⚽ Football News Assistant

A smart, multilingual football news assistant that retrieves the latest football news, summarizes articles, and answers user questions in natural language.
Supports translation, semantic search, and real-time responses using modern NLP and AI tools.

🔹 Features

Retrieve football news from major leagues: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League.

Semantic search using SentenceTransformer + FAISS to find relevant articles.

Summarize and answer questions using Mistral AI.

Multilingual support with automatic translation via mBART.

Sources displayed in bold for clarity.

Interactive web interface with live chat.

⚽ Loading spinner while processing requests.

🔹 RAG (Retrieval-Augmented Generation) Pipeline

The assistant implements a RAG pipeline combining retrieval and generation:

1️⃣ Retrieval (R)

Fetch news articles from NewsAPI.

Convert articles into embeddings with SentenceTransformer.

Store embeddings in a FAISS index for fast similarity search.

Encode the user query and retrieve the most relevant articles.

2️⃣ Augmented Generation (AG)

Format retrieved articles as context for the AI model.

Mistral generates a concise, factual answer.

If the query is not in English, it is translated automatically to English before retrieval.

3️⃣ Output

The generated answer is translated back (if needed) to the user’s language.

Sources are displayed in bold.

The ⚽ spinner is shown while the assistant is generating a response.

🔹 RAG Pipeline Diagram

1.User Query (any language)

2.Language Detection

3.Translation to English (if needed)

4.Embed Query with SentenceTransformer

5.FAISS Vector Search

6.Retrieve Top Articles

7.Mistral Generates Summary

8.Translation to User Language (if needed)

9.Display Response with sources in bold


🔹 Tech Stack

Backend: Flask

Frontend: HTML, CSS, JavaScript

News Retrieval: NewsAPI

NLP & Embeddings:

SentenceTransformer (all-MiniLM-L6-v2)

FAISS for vector search

Language Detection: langdetect

Translation: MBart50 (HuggingFace Transformers)

AI Assistant: Mistral API (mistral-small-latest)

Data Handling: Pandas, NumPy

🔹 File Structure


<img width="776" height="313" alt="image" src="https://github.com/user-attachments/assets/d60f85ca-429b-4139-8fd8-456fc3ad28e5" />

