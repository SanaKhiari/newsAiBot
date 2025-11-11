import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from langdetect import detect
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import faiss, numpy as np, pandas as pd
from datetime import datetime, timedelta

load_dotenv()  # this will load variables from .env into environment

# === Configuration ===
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not MISTRAL_API_KEY or not NEWS_API_KEY:
    raise RuntimeError("Missing API keys. Please set MISTRAL_API_KEY and NEWS_API_KEY in your environment or .env file.")
client = Mistral(api_key=MISTRAL_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Traduction via mBART ===
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model_trans = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate_text(text, src_lang, tgt_lang):
    lang_code_map = {
        "en": "en_XX", "fr": "fr_XX", "es": "es_XX", 
        "de": "de_DE", "ar": "ar_AR", "it": "it_IT", "pt": "pt_XX"
    }
    src_code = lang_code_map.get(src_lang, "en_XX")
    tgt_code = lang_code_map.get(tgt_lang, "en_XX")
    tokenizer.src_lang = src_code
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    generated_tokens = model_trans.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
        max_length=512
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# === Récupération des news ===
def fetch_football_news(days=2):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    queries = ["Champions League", "Premier League", "La Liga", "Serie A", "Ligue 1", "Bundesliga"]
    all_articles = []

    for q in queries:
        res = newsapi.get_everything(
            q=q, language="en", sort_by="publishedAt", 
            from_param=from_date, to=to_date, page_size=50
        )
        for a in res.get("articles", []):
            if a.get("title") and a.get("description"):
                all_articles.append({
                    "title": a["title"],
                    "description": a["description"],
                    "source": a["source"]["name"],  # seulement le nom de la source
                    "publishedAt": a["publishedAt"]
                })

    df = pd.DataFrame(all_articles)

    # Calcul des embeddings
    embeddings = model.encode(
        (df["title"] + ". " + df["description"]).tolist(),
        show_progress_bar=False
    )
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return df, index, embeddings

# === Récupération par similarité ===
def retrieve_articles(query, df, index, top_k=15, team=None):
    query_vec = model.encode([query])
    _, indices = index.search(query_vec, top_k)
    results = [df.iloc[i].to_dict() for i in indices[0]]
    if team:
        results = [a for a in results if team.lower() in (a['title'] + " " + a['description']).lower()]
    return results

# === Génération de réponse ===
def generate_response_from_articles(query, articles):
    if not articles:
        return "No recent updates available."

    query_lang = detect(query)
    query_en = translate_text(query, query_lang, "en") if query_lang != "en" else query

    # Concaténation des articles avec la source en gras
    context = "\n\n".join([
        f"{a['title']} - {a.get('description','')[:250]} (<b>{a.get('source','N/A')}</b>)"
        for a in articles
    ])

    prompt = f"""
    You are a professional football journalist. 
    Based on the following recent articles, write a short and factual summary (2–3 sentences) 
    that directly answers the user's question. Do not invent information. Mention the sources in parentheses.
    
    Articles: {context}
    Question: {query_en}
    Answer:
    """

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=250
    )

    answer_en = response.choices[0].message.content.strip()
    return translate_text(answer_en, "en", query_lang) if query_lang != "en" else answer_en
