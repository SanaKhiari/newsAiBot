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
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler  # ✅ Modèle plus simple
from transformers import CLIPProcessor, CLIPModel
import gc
import numpy as np

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.environ.get("SANAapiKey")
groq_client = Groq(api_key=GROQ_API_KEY)
BEST_MODEL = "llama-3.3-70b-versatile"

# ✅ XAI Configuration
XAI_CONFIG = {
    "similarity_threshold": 0.60,  # Seuil abaissé
    "max_retries": 3,              # Maximum 3 essais
    "show_explanations": True,     # Activer XAI
    "save_debug_info": True        # Sauvegarder infos debug
}

# Dossiers de sortie
OUTPUT_DIR = "generated_outputs"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
XAI_DIR = os.path.join(OUTPUT_DIR, "xai_explanations")  # ✅ Nouveau dossier

for directory in [OUTPUT_DIR, IMAGE_DIR, JSON_DIR, XAI_DIR]:
    os.makedirs(directory, exist_ok=True)

# Variables globales pour les modèles
_sd_pipe = None  # ✅ SD 2.1 au lieu de SDXL
_clip_model = None
_clip_processor = None

def get_sd_pipe():
    """Charge Stable Diffusion 2.1 (plus léger que SDXL)"""
    global _sd_pipe
    if _sd_pipe is None:
        print("🎨 Chargement de Stable Diffusion 2.1 (modèle léger)...")
        MODEL_ID = "stabilityai/stable-diffusion-2-1"  # ✅ Plus rapide que SDXL
        
        _sd_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None  # Désactiver pour vitesse
        )
        _sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(_sd_pipe.scheduler.config)
        _sd_pipe = _sd_pipe.to("cpu")
        _sd_pipe.enable_attention_slicing()
        print("✅ SD 2.1 chargé (plus rapide que SDXL)")
    return _sd_pipe

def get_clip_models():
    """Charge CLIP une seule fois"""
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        print("🔍 Chargement de CLIP...")
        MODEL_ID_CLIP = "openai/clip-vit-base-patch32"
        
        _clip_model = CLIPModel.from_pretrained(MODEL_ID_CLIP)
        _clip_processor = CLIPProcessor.from_pretrained(MODEL_ID_CLIP)
        _clip_model = _clip_model.to("cpu")
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
    """Agent 1: Génère le résumé"""
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

def agent_social_generator(article_text, summary, post_type, attempt=1):
    """Agent 2: Génère le post social (avec numéro de tentative)"""
    print(f"\n🤖 AGENT 2 - SOCIAL POST GENERATOR ({post_type.upper()}) - Attempt {attempt}")
    
    article_text = clean_text(article_text)
    summary = clean_text(summary)
    
    # ✅ Température variable pour diversité
    temperature = 0.7 + (attempt - 1) * 0.1  # 0.7, 0.8, 0.9 pour tentatives 1, 2, 3
    
    if post_type == "twitter":
        tweet_prompt = f"""Instruction:
Generate a concise and engaging tweet (≤280 characters) from the summary below. Include at least 1 emoji and 1–2 hashtags.

Summary: {summary}
"""
        
        tweet = generate_with_groq(tweet_prompt, temperature=temperature, max_tokens=150)
        tweet = re.sub(r'^(Tweet:|Post:|Here\'s a tweet:|Here is a tweet:)\s*', '', tweet, flags=re.IGNORECASE)
        tweet = tweet.strip('"').strip()
        
        result = {
            "post_type": "twitter",
            "tweet": tweet,
            "article": article_text,
            "summary": summary,
            "attempt": attempt,
            "temperature": temperature
        }
        
        print(f"✅ Tweet generated ({len(tweet)} characters)")
        return result
    
    elif post_type == "instagram":
        insta_prompt = f"""Instruction:
Generate a short, engaging Instagram caption (1 sentence) and suggest a corresponding image prompt. Include at least 1 emoji.

Output format:
- Single sentence caption
- At least 1 emoji
- Image prompt clearly separated with 'Sugg-image:'
- Image prompt MUST be maximum 40 words

Summary: {summary}
"""
        
        insta_response = generate_with_groq(insta_prompt, temperature=temperature, max_tokens=400)
        
        # Parsing
        parts = insta_response.split('Sugg-image:', 1)
        if len(parts) == 2:
            caption = parts[0].strip().strip('"').strip()
            image_prompt_raw = parts[1].strip().strip('"').strip("'").strip()
            
            # Troncature à 40 mots (plus strict pour SD 2.1)
            image_prompt_words = image_prompt_raw.split()
            if len(image_prompt_words) > 40:
                image_prompt = ' '.join(image_prompt_words[:40])
                print(f"⚠️ Image prompt truncated: {len(image_prompt_words)} → 40 words")
            else:
                image_prompt = image_prompt_raw
        else:
            caption = insta_response.strip()
            image_prompt = "modern digital illustration, vibrant colors, professional photography"
            print(f"⚠️ Fallback: Format not respected")
        
        result = {
            "post_type": "instagram",
            "caption": caption,
            "image_prompt": image_prompt,
            "article": article_text,
            "summary": summary,
            "attempt": attempt,
            "temperature": temperature
        }
        
        print(f"✅ Instagram caption generated (attempt {attempt})")
        print(f"✅ Image prompt: {image_prompt}")
        
        return result

def agent_image_generator(image_prompt, output_path, attempt=1):
    """Agent 3: Génère l'image avec SD 2.1 (plus rapide)"""
    print(f"\n🤖 AGENT 3 - IMAGE GENERATOR - Attempt {attempt}")
    print(f"🎨 Prompt: {image_prompt[:100]}...")
    
    try:
        gc.collect()
        
        # Charger SD 2.1
        pipe = get_sd_pipe()
        
        negative_prompt = "blurry, low quality, distorted, ugly, duplicate"
        
        print("⏳ Generating image (SD 2.1 - faster than SDXL, ~60-90 sec)...")
        
        # ✅ Seed variable pour diversité
        seed = 42 + (attempt - 1) * 100
        
        image = pipe(
            prompt=image_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,  # ✅ 25 steps pour SD 2.1
            guidance_scale=7.5,
            height=512,  # ✅ 512x512 pour SD 2.1 (plus rapide)
            width=512,
            generator=torch.manual_seed(seed)
        ).images[0]
        
        image.save(output_path)
        
        print(f"✅ Image generated (seed={seed}, 512x512, attempt {attempt})")
        print(f"📁 Saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        return None

def compute_clip_similarity_with_xai(image_path, text_prompt, attempt=1):
    """
    ✅ XAI VERSION: Calcule la similarité + explications détaillées
    """
    print(f"\n🔍 Computing CLIP similarity (Attempt {attempt})...")
    
    try:
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
            
            # ✅ XAI: Logits bruts pour analyse
            logits_per_image = outputs.logits_per_image[0][0].item()
        
        # ✅ XAI: Interprétation détaillée
        xai_explanation = generate_xai_explanation(
            similarity_normalized, 
            text_prompt_truncated,
            attempt
        )
        
        print(f"📊 Raw logits: {logits_per_image:.4f}")
        print(f"📊 Cosine similarity: {cosine_sim:.4f}")
        print(f"📊 Normalized score: {similarity_normalized:.4f}")
        print(f"📊 XAI Interpretation: {xai_explanation['interpretation']}")
        
        return {
            'similarity_score': similarity_normalized,
            'cosine_sim': cosine_sim,
            'logits': logits_per_image,
            'xai_explanation': xai_explanation,
            'attempt': attempt
        }
        
    except Exception as e:
        print(f"❌ CLIP Error: {str(e)}")
        return {
            'similarity_score': 0.0,
            'error': str(e),
            'attempt': attempt
        }

def generate_xai_explanation(score, prompt, attempt):
    """
    ✅ XAI: Génère une explication humaine du score CLIP
    """
    if score >= 0.75:
        interpretation = "Excellent alignment"
        color = "green"
        recommendation = "Image perfectly matches the caption. Ready to publish!"
    elif score >= 0.65:
        interpretation = "Good alignment"
        color = "blue"
        recommendation = "Image matches well. Acceptable for publication."
    elif score >= 0.55:
        interpretation = "Moderate alignment"
        color = "orange"
        recommendation = "Image somewhat matches. Consider regenerating for better quality."
    elif score >= 0.45:
        interpretation = "Weak alignment"
        color = "red"
        recommendation = "Image doesn't match well. Regeneration recommended."
    else:
        interpretation = "Poor alignment"
        color = "darkred"
        recommendation = "Image is incoherent with caption. Must regenerate."
    
    # ✅ XAI: Analyse des mots clés
    keywords = extract_keywords(prompt)
    
    return {
        'score': score,
        'interpretation': interpretation,
        'color': color,
        'recommendation': recommendation,
        'keywords_detected': keywords,
        'attempt': attempt,
        'threshold_status': 'PASSED' if score >= XAI_CONFIG['similarity_threshold'] else 'FAILED'
    }

def extract_keywords(text):
    """
    ✅ XAI: Extrait les mots-clés du prompt (simulation simple)
    """
    # Mots vides à ignorer
    stopwords = {'a', 'an', 'the', 'with', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
    words = text.lower().split()
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    return keywords[:5]  # Top 5 keywords

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
    """
    ✅ ROUTE PRINCIPALE AVEC BOUCLE DE RÉGÉNÉRATION (MAX 3 ESSAIS)
    """
    try:
        # Récupérer les paramètres
        post_type = request.form.get("post_type", "twitter")
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
        print(f"🚀 STARTING {post_type.upper()} POST GENERATION WITH XAI")
        print(f"{'='*80}")
        
        # Phase 1: Résumé
        summary = agent_summarizer(article_text)
        
        # ✅ BOUCLE DE RÉGÉNÉRATION (MAX 3 ESSAIS)
        attempts_history = []
        final_result = None
        best_score = 0.0
        
        for attempt in range(1, XAI_CONFIG['max_retries'] + 1):
            print(f"\n{'='*80}")
            print(f"🔄 ATTEMPT {attempt}/{XAI_CONFIG['max_retries']}")
            print(f"{'='*80}")
            
            # Phase 2: Génération du post
            social_result = agent_social_generator(article_text, summary, post_type, attempt=attempt)
            
            if post_type == "twitter":
                # Twitter: pas d'image, on retourne direct
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
                image_filename = f"instagram_attempt{attempt}_{timestamp}.png"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                
                # Phase 3: Génération image
                generated_path = agent_image_generator(social_result['image_prompt'], image_path, attempt=attempt)
                
                if not generated_path:
                    print(f"❌ Image generation failed (attempt {attempt})")
                    attempts_history.append({
                        'attempt': attempt,
                        'status': 'failed',
                        'error': 'Image generation failed'
                    })
                    continue
                
                # Phase 4: Validation CLIP avec XAI
                clip_result = compute_clip_similarity_with_xai(
                    generated_path, 
                    social_result['image_prompt'],
                    attempt=attempt
                )
                
                similarity_score = clip_result['similarity_score']
                xai_explanation = clip_result.get('xai_explanation', {})
                
                # Enregistrer tentative
                attempt_data = {
                    'attempt': attempt,
                    'caption': social_result['caption'],
                    'image_prompt': social_result['image_prompt'],
                    'image_filename': image_filename,
                    'similarity_score': similarity_score,
                    'xai_explanation': xai_explanation,
                    'temperature': social_result['temperature'],
                    'timestamp': timestamp
                }
                attempts_history.append(attempt_data)
                
                # Mettre à jour le meilleur score
                if similarity_score > best_score:
                    best_score = similarity_score
                    final_result = attempt_data
                
                # ✅ Vérifier si le seuil est atteint
                if similarity_score >= XAI_CONFIG['similarity_threshold']:
                    print(f"\n{'='*80}")
                    print(f"✅ SUCCESS: Similarity {similarity_score:.4f} >= {XAI_CONFIG['similarity_threshold']}")
                    print(f"✅ Validated after {attempt} attempt(s)")
                    print(f"{'='*80}")
                    break
                else:
                    print(f"\n{'='*80}")
                    print(f"⚠️ Attempt {attempt}: Score {similarity_score:.4f} < {XAI_CONFIG['similarity_threshold']}")
                    if attempt < XAI_CONFIG['max_retries']:
                        print(f"🔄 Regenerating with higher temperature...")
                    print(f"{'='*80}")
        
        # ✅ Résultat final (meilleur score ou dernier essai)
        if not final_result and attempts_history:
            final_result = attempts_history[-1]
        
        if not final_result:
            return jsonify({"error": "All generation attempts failed"}), 500
        
        # Sauvegarde JSON avec historique XAI
        json_filename = f"instagram_{timestamp}_xai.json"
        json_path = os.path.join(JSON_DIR, json_filename)
        
        result_data = {
            'status': 'success' if final_result['similarity_score'] >= XAI_CONFIG['similarity_threshold'] else 'below_threshold',
            'post_type': 'instagram',
            'article': article_text,
            'summary': summary,
            'final_result': final_result,
            'attempts_history': attempts_history,
            'xai_config': XAI_CONFIG,
            'best_score': best_score,
            'total_attempts': len(attempts_history)
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # ✅ Sauvegarder explication XAI séparée
        if XAI_CONFIG['save_debug_info']:
            xai_filename = f"xai_explanation_{timestamp}.json"
            xai_path = os.path.join(XAI_DIR, xai_filename)
            with open(xai_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'attempts': attempts_history,
                    'threshold': XAI_CONFIG['similarity_threshold'],
                    'final_decision': final_result['xai_explanation']
                }, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "status": result_data['status'],
            "post_type": "instagram",
            "caption": final_result['caption'],
            "summary": summary,
            "image_filename": final_result['image_filename'],
            "similarity_score": round(final_result['similarity_score'], 4),
            "is_valid": final_result['similarity_score'] >= XAI_CONFIG['similarity_threshold'],
            "threshold": XAI_CONFIG['similarity_threshold'],
            "xai_explanation": final_result['xai_explanation'],
            "total_attempts": len(attempts_history),
            "attempts_history": attempts_history,
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