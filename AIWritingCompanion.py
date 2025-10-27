import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq
import PyPDF2



# Load environment variables
load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.environ.get("SANAapiKey"))

# Helper function to split text into manageable chunks
def split_text(text, max_tokens=100000): #modèle peut prendre en compte jusqu'à 128 000 tokens dans une seule requête
    words = text.split()
    chunks = []
    chunk = []
    count = 0
    for word in words:
        count += 1
        chunk.append(word)
        if count >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
            count = 0
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks




def chat():
    # ✅ Si l'utilisateur arrive sur /chat via un clic (GET)
    if request.method == "GET":
        return render_template("AIWritingCompanion.html")  # affiche la page du chat

    # --- Si c'est un POST (soumission du formulaire ou requête AJAX) ---
    user_message = request.form.get("message")
    style = request.form.get("style")
    tone = request.form.get("tone")
    mode = request.form.get("mode")

    # Hyperparameters from user input or defaults
    temperature = float(request.form.get("temperature", 0.7))
    max_tokens = int(request.form.get("max_tokens", 15000))
    top_p = float(request.form.get("top_p", 0.9))

    # Handle PDF upload
    pdf_file = request.files.get("pdf_file")
    pdf_text = ""
    if pdf_file and pdf_file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"

    # Combine user message and PDF content
    combined_text = user_message or ""
    if pdf_text:
        combined_text += f"\n\nPDF content:\n{pdf_text}"

    if not combined_text.strip():
        return jsonify({"response": "Please enter a message or upload a PDF."})

    # Split large text into chunks
    chunks = split_text(combined_text, max_tokens=100000)

    responses = []
    for chunk in chunks:
        prompt = f"""
            You are an expert, advanced AI writing assistant, specialized in creative and strategic text transformation.

            Your task is to execute the following request with impeccable quality:
            Task: {mode}
            Desired Writing Style: {style}
            Target Tone: {tone}

            The user provided this initial text (the source material):
            "{chunk}"

            ### Instructions for High-Quality Output:
            1. **Perspective:** Rewrite or expand the text strictly in the **first person** (I/We).
            2. **Quality & Fluency:** The final output must be **highly fluent, natural, and engaging**. Ensure the language is polished and reads as if written by a skilled human. **Avoid awkward phrasing, jargon, or repetitions.**
            3. **Meaning Preservation:** Maintain the core message and essential facts from the original text.
            4. **Style Execution:** Apply the specified '{style}' and '{tone}' with complete consistency throughout the entire text.
            5. **Format:** Respond **only** with the complete, rewritten or generated text. Do not include any titles, explanations, greetings, or meta-comments.
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        responses.append(chat_completion.choices[0].message.content)

    final_response = "\n\n".join(responses)

    # Étape 2 : évaluation du texte par le modèle juge
    judge_prompt = f"""
    You are an impartial evaluator.

    Your task is to check whether the following AI-generated answer correctly follows the user’s instructions and fulfills the request.

    Evaluate:
    1. Does the answer address all parts of the original task?
    2. Does it respect the desired style and tone?
    3. Does it preserve the meaning of the source text?
    4. Is it fluent, natural, and error-free?

    Return your evaluation as plain text (not JSON).

    Original Task: {mode}, Style: {style}, Tone: {tone}
    Original Text: "{combined_text}"
    Generated Answer: "{final_response}"
    """

    judge_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": judge_prompt}],
        model="moonshotai/kimi-k2-instruct-0905",
        temperature=0,
        max_tokens=2000
    )

    print("\n=== Judge Evaluation ===")
    print(judge_completion.choices[0].message.content, flush=True)
    print("========================\n", flush=True)

    return jsonify({"response": final_response})



