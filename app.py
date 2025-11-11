from flask import Flask, render_template, request, jsonify
from mistral_helper import fetch_football_news, retrieve_articles, generate_response_from_articles

app = Flask(__name__)

# Charger les données une seule fois au démarrage
football_df, index, embeddings = fetch_football_news()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")

    # Step 1: retrieve relevant articles (same as in Colab)
    articles = retrieve_articles(user_query, football_df, index, top_k=10)

    # Step 2: generate response using the exact same logic as Colab
    answer = generate_response_from_articles(user_query, articles)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
