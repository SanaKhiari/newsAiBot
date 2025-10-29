import os
import re
import json
import torch
from datetime import datetime
from flask import request, jsonify, send_file
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel
import gc
import io

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.environ.get("SANAapiKey")
groq_client = Groq(api_key=GROQ_API_KEY)
BEST_MODEL = "llama-3.3-70b-versatile"

# Dossiers de sortie
OUTPUT_DIR = "generated_outputs"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

for directory in [OUTPUT_DIR, IMAGE_DIR, JSON_DIR]:
    os.makedirs(directory, exist_ok=True)

# Variables globales pour les modèles (chargement lazy)
_sdxl_pipe = None
_clip_model = None
_clip_processor = None

def get_sdxl_pipe():
    """Charge SDXL une seule fois (lazy loading)"""
    global _sdxl_pipe
    if _sdxl_pipe is None:
        print("🎨 Chargement de Stable Diffusion XL...")
        MODEL_ID_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
        
        _sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID_SDXL,
            torch_dtype=torch.float16,  # CPU = float32
            use_safetensors=True
        )
        _sdxl_pipe.scheduler = DPMSolverMultistepScheduler.from_config(_sdxl_pipe.scheduler.config)
        _sdxl_pipe.enable_attention_slicing()
        _sdxl_pipe.enable_model_cpu_offload()
        _sdxl_pipe = _sdxl_pipe.to("cuda")
        print("✅ SDXL chargé")
    return _sdxl_pipe

def get_clip_models():
    """Charge CLIP une seule fois (lazy loading)"""
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        print("🔍 Chargement de CLIP...")
        MODEL_ID_CLIP = "openai/clip-vit-base-patch32"
        
        _clip_model = CLIPModel.from_pretrained(MODEL_ID_CLIP)
        _clip_processor = CLIPProcessor.from_pretrained(MODEL_ID_CLIP)
        _clip_model = _clip_model.to("cuda")
        _clip_model.eval()
        print("✅ CLIP chargé")
    return _clip_model, _clip_processor

def clean_text(text):
    """Nettoie le texte"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def generate_with_groq(prompt, temperature=0.7, max_tokens=300):
    """Appel API Groq"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert social media content creator. Always respond in English."},
                {"role": "user", "content": prompt}
            ],
            model=BEST_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error API: {str(e)}"

def agent_summarizer(article_text):
    """Agent 1: Génère le résumé de l'article en ANGLAIS"""
    print("\n🤖 AGENT 1 - SUMMARIZER")
    
    article_text = clean_text(article_text)
    
    prompt = f"""You are an expert in creating social media content.

TASK: Summarize the following article in 3-4 concise and engaging sentences.

GUIDELINES:
- Keep essential information
- Engaging and accessible style
- Maximum 80 words
- No introductory formulas
- RESPOND IN ENGLISH

ARTICLE:
{article_text}

SUMMARY:"""
    
    summary = generate_with_groq(prompt)
    print(f"✅ Summary generated ({len(summary.split())} words)")
    return summary

def agent_social_generator(article_text, summary, post_type):
    """Agent 2: Génère le post social selon le type"""
    print(f"\n🤖 AGENT 2 - SOCIAL POST GENERATOR ({post_type.upper()})")
    
    article_text = clean_text(article_text)
    summary = clean_text(summary)
    
    if post_type == "twitter":
        # Prompt Twitter
        tweet_prompt = f"""Instruction:
Generate a concise and engaging tweet (≤280 characters) from the summary below. Include at least 1 emoji and 1–2 hashtags. Capture the most important idea and make it easy to understand for a general audience.

Context:
You are a social media assistant. The goal is to inform and engage readers with short, appealing messages.

Output format:
- Maximum 280 characters
- 1–2 hashtags
- At least 1 emoji
- Single concise sentence

Summary: {summary}
"""
        
        tweet = generate_with_groq(tweet_prompt, temperature=0.8, max_tokens=150)
        tweet = re.sub(r'^(Tweet:|Post:|Here\'s a tweet:|Here is a tweet:)\s*', '', tweet, flags=re.IGNORECASE)
        tweet = tweet.strip('"').strip()
        
        result = {
            "post_type": "twitter",
            "tweet": tweet,
            "article": article_text,
            "summary": summary
        }
        
        print(f"✅ Tweet generated ({len(tweet)} characters)")
        return result
    
    elif post_type == "instagram":
        # Prompt Instagram
        insta_prompt = f"""Instruction:
Generate a short, engaging Instagram caption (1 sentence) from the summary below and suggest a corresponding image prompt. Include at least 1 emoji. The caption should be appealing and easy to read.

Context:
You are a social media assistant. The goal is to attract attention and create visually appealing posts.

Output format:
- Single sentence caption
- At least 1 emoji
- Image prompt clearly separated with 'Sugg-image:'
- Image prompt MUST be maximum 50 words

Summary: {summary}
"""
        
        insta_response = generate_with_groq(insta_prompt, temperature=0.8, max_tokens=400)
        
        # Parsing
        parts = insta_response.split('Sugg-image:', 1)
        if len(parts) == 2:
            caption = parts[0].strip().strip('"').strip()
            image_prompt_raw = parts[1].strip().strip('"').strip("'").strip()
            
            # Troncature à 50 mots
            image_prompt_words = image_prompt_raw.split()
            if len(image_prompt_words) > 50:
                image_prompt = ' '.join(image_prompt_words[:50])
                print(f"⚠️ Image prompt truncated: {len(image_prompt_words)} → 50 words")
            else:
                image_prompt = image_prompt_raw
        else:
            caption = insta_response.strip()
            image_prompt = "modern digital illustration, vibrant colors, professional photography, high quality"
            print(f"⚠️ Fallback: Format not respected, using default prompt")
        
        result = {
            "post_type": "instagram",
            "caption": caption,
            "image_prompt": image_prompt,
            "article": article_text,
            "summary": summary
        }
        
        print(f"✅ Instagram caption generated")
        print(f"✅ Image prompt: {image_prompt}")
        
        return result

def agent_image_generator(image_prompt, output_path):
    """Agent 3: Génère l'image avec SDXL"""
    print(f"\n🤖 AGENT 3 - IMAGE GENERATOR")
    print(f"🎨 Prompt: {image_prompt[:150]}...")
    
    try:
        # Libérer mémoire
        gc.collect()
        
        # Charger SDXL
        pipe = get_sdxl_pipe()
        
        negative_prompt = "blurry, low quality, distorted"
        
        print("⏳ Generating image (cela peut prendre 2-3 minutes sur CPU)...")
        
        image = pipe(
            prompt=image_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=torch.manual_seed(42)
        ).images[0]
        
        image.save(output_path)
        
        print(f"✅ Image generated and saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        return None

def compute_clip_similarity(image_path, text_prompt):
    """Calcule la similarité sémantique CLIP"""
    print(f"\n🔍 Computing CLIP similarity...")
    
    try:
        # Charger CLIP
        clip_model, clip_processor = get_clip_models()
        
        # Charger image
        image = Image.open(image_path).convert("RGB")
        
        # Troncature du prompt
        tokens = clip_processor.tokenizer(
            text_prompt,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        prompt_length = tokens['input_ids'].shape[1]
        if prompt_length >= 77:
            print(f"⚠️ Prompt truncated: {prompt_length} → 77 tokens")
            text_prompt_truncated = clip_processor.tokenizer.decode(
                tokens['input_ids'][0][:77],
                skip_special_tokens=True
            )
        else:
            text_prompt_truncated = text_prompt
        
        # Préparer inputs
        inputs = clip_processor(
            text=[text_prompt_truncated],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        
        # Calcul de similarité
        with torch.no_grad():
            outputs = clip_model(**inputs)
            
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalisation
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            cosine_sim = (image_embeds @ text_embeds.T).squeeze().item()
            similarity_normalized = (cosine_sim + 1) / 2
        
        print(f"📊 Cosine similarity: {cosine_sim:.4f}")
        print(f"📊 Normalized score: {similarity_normalized:.4f}")
        
        return similarity_normalized
        
    except Exception as e:
        print(f"❌ CLIP Error: {str(e)}")
        return 0.0

def extract_text_from_pdf(pdf_file):
    """Extrait le texte d'un PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        return ""

def social_media_post():
    """Route principale pour générer des posts sociaux"""
    try:
        # Récupérer les paramètres
        post_type = request.form.get("post_type", "twitter")  # twitter ou instagram
        user_text = request.form.get("article_text", "").strip()
        pdf_file = request.files.get("pdf_file")
        
        # Extraction du texte
        article_text = ""
        
        if pdf_file and pdf_file.filename.endswith(".pdf"):
            print("📄 PDF détecté, extraction en cours...")
            article_text = extract_text_from_pdf(pdf_file)
            if not article_text:
                return jsonify({"error": "Failed to extract text from PDF"}), 400
        elif user_text:
            article_text = user_text
        else:
            return jsonify({"error": "Please provide article text or upload a PDF"}), 400
        
        if len(article_text) < 50:
            return jsonify({"error": "Article text is too short (minimum 50 characters)"}), 400
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING {post_type.upper()} POST GENERATION")
        print(f"{'='*80}")
        
        # Phase 1: Résumé
        summary = agent_summarizer(article_text)
        
        # Phase 2: Génération du post
        social_result = agent_social_generator(article_text, summary, post_type)
        
        if post_type == "twitter":
            # Twitter: pas d'image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"twitter_{timestamp}.json"
            json_path = os.path.join(JSON_DIR, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(social_result, f, ensure_ascii=False, indent=2)
            
            return jsonify({
                "status": "success",
                "post_type": "twitter",
                "tweet": social_result['tweet'],
                "summary": summary,
                "json_file": json_filename
            })
        
        elif post_type == "instagram":
            # Instagram: génération d'image + validation CLIP
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"instagram_{timestamp}.png"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            
            # Phase 3: Génération image
            generated_path = agent_image_generator(social_result['image_prompt'], image_path)
            
            if not generated_path:
                return jsonify({"error": "Image generation failed"}), 500
            
            # Phase 4: Validation CLIP
            similarity_score = compute_clip_similarity(generated_path, social_result['image_prompt'])
            
            threshold = 0.65  # Seuil plus souple pour CPU
            is_valid = similarity_score >= threshold
            
            # Sauvegarde JSON
            json_filename = f"instagram_{timestamp}.json"
            json_path = os.path.join(JSON_DIR, json_filename)
            
            result_data = {
                **social_result,
                "image_filename": image_filename,
                "similarity_score": similarity_score,
                "is_valid": is_valid,
                "threshold": threshold,
                "timestamp": timestamp
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            return jsonify({
                "status": "success",
                "post_type": "instagram",
                "caption": social_result['caption'],
                "summary": summary,
                "image_filename": image_filename,
                "similarity_score": round(similarity_score, 4),
                "is_valid": is_valid,
                "threshold": threshold,
                "json_file": json_filename
            })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def download_image():
    """Route pour télécharger l'image générée"""
    try:
        image_filename = request.args.get("filename")
        if not image_filename:
            return jsonify({"error": "No filename provided"}), 400
        
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"}), 404
        
        return send_file(
            image_path,
            mimetype='image/png',
            as_attachment=True,
            download_name=image_filename
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500