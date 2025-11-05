import os
from flask import Flask, render_template
from dotenv import load_dotenv
from groq import Groq
import PyPDF2

from AIWritingCompanion import chat
from SocialMediaGenerator import social_media_post, download_image

app = Flask(__name__)

# ------------ PAGE D'ACCUEIL ------------
@app.route("/")
def home():
    return render_template("home.html")

# ------------ PAGE CHAT AI ------------
app.add_url_rule("/chat", view_func=chat, methods=["GET", "POST"], endpoint="chat")

# ------------ PAGE SOCIAL MEDIA GENERATOR ------------
@app.route("/social-media", methods=["GET"])
def social_media_page():
    """Affiche la page de génération de posts sociaux"""
    return render_template("social_media.html")

@app.route("/generate-post", methods=["POST"])
def generate_post():
    """API pour générer un post social"""
    return social_media_post()

@app.route("/download-image", methods=["GET"])
def download_image_route():
    """API pour télécharger l'image générée"""
    return download_image()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)