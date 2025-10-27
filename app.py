import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq
import PyPDF2

from AIWritingCompanion import chat  # ✅ importer ta fonction depuis le module



app = Flask(__name__)
# ------------ PAGE D’ACCUEIL ------------
@app.route("/")
def home():
    return render_template("home.html")  # page d’accueil par défaut

# ------------ PAGE CHAT AI ------------
# On associe la fonction chat() importée à la route /chat
app.add_url_rule("/chat", view_func=chat, methods=["GET", "POST"], endpoint="chat")


if __name__ == "__main__":
    app.run(debug=True)
