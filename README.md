# AI Writing Companion 🤖✍️

![Flask](https://img.shields.io/badge/Flask-2.3.3-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Groq API](https://img.shields.io/badge/Groq-API-orange)

Une application web intelligente d'assistant à l'écriture propulsée par l'IA, capable de transformer et d'améliorer vos textes selon votre style et ton préférés.

---

## 🌟 Fonctionnalités

- 🔄 **Transformation de texte intelligente** – Réécriture et amélioration de textes
- 📄 **Traitement de PDF** – Extraction et analyse de documents PDF
- 🎨 **Personnalisation avancée** – Style, ton, mode d'écriture configurables
- ⚡ **Évaluation automatique** – Système d'auto-évaluation de la qualité
- 📝 **Gestion de longs documents** – Découpage automatique des textes volumineux

---

## 🛠️ Technologies Utilisées

- **Backend :** Flask, Python  
- **IA :** Groq API (Llama 3.3 70B, Moonshot Kimi)  
- **Traitement PDF :** PyPDF2  
- **Sécurité :** python-dotenv  
- **Frontend :** HTML, JavaScript, AJAX  

---

## 📦 Installation

### Prérequis

- Python 3.8+
- Clé API Groq

### Étapes d'installation

1. **Créer un environnement virtuel**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```
2. **Installer les dépendances**
```bash

pip install -r requirements.txt
```
3. **Configurer les variables d'environnement**
```bash
cp .env.example .env
```
## Modèles IA Utilisés

Génération : llama-3.3-70b-versatile

Évaluation (llm as a judge) : moonshotai/kimi-k2-instruct-0905

## Préparation du prompt

Chaque chunk est transformé en prompt structuré pour guider le modèle :

* Instructions sur le style, le ton, le mode.

* Règles pour préserver la qualité, la fluidité et le sens.

### Technique utilisée : prompt engineering pour obtenir un texte de haute qualité.

##  Évaluation automatique – modèle juge

Après génération, un second modèle évalue le texte produit :

Respect du style et du ton.

Exactitude et fidélité au texte source.

Qualité linguistique.

Cela correspond à une forme de self-assessment ou QA automatisée.
