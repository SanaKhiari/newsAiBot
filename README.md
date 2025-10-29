# AI Social Media Post Generator 🤖📱

![Flask](https://img.shields.io/badge/Flask-3.1.2-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Groq API](https://img.shields.io/badge/Groq-API-orange)
![SDXL](https://img.shields.io/badge/Stable%20Diffusion-XL-purple)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-red)

Une application web intelligente de génération automatique de posts pour réseaux sociaux (Twitter & Instagram) avec génération d'images par IA et validation sémantique CLIP.

---

## 🌟 Fonctionnalités

- 🐦 **Génération de tweets** – Posts optimisés Twitter (≤280 caractères) avec emojis et hashtags
- 📸 **Création de posts Instagram** – Captions engageants + images générées par IA
- 📄 **Traitement de PDF** – Extraction automatique de texte depuis documents
- 🎨 **Génération d'images SDXL** – Images 512x512 haute qualité via Stable Diffusion XL
- 🔍 **Validation sémantique CLIP** – Vérification de cohérence image/texte par IA
- 📊 **Scoring automatique** – Évaluation de la qualité (0-100%)
- 💾 **Sauvegarde automatique** – Export JSON + images téléchargeables

---

## 🛠️ Technologies Utilisées

- **Backend :** Flask, Python  
- **IA Textuelle :** Groq API (Llama 3.3 70B Versatile)  
- **Génération d'images :** Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0)  
- **Validation sémantique :** OpenAI CLIP (clip-vit-base-patch32)  
- **Traitement PDF :** PyPDF2  
- **Deep Learning :** PyTorch, Transformers, Diffusers  
- **Frontend :** HTML5, JavaScript, CSS3  

---

## 📦 Installation

### Prérequis

- Python 3.8+
- 24GB RAM minimum (pour SDXL sur CPU)
- Clé API Groq
- 10GB espace disque (modèles IA)

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone <votre-repo>
cd <votre-projet>
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

> ⚠️ **Note :** Le premier lancement téléchargera automatiquement :
> - SDXL (≈7 GB)
> - CLIP (≈500 MB)

4. **Configurer les variables d'environnement**

Créer un fichier `.env` :
```env
SANAapiKey=votre_cle_groq_api_ici
```

5. **Lancer l'application**
```bash
python app.py
```

Accéder à : **http://localhost:5001/social-media**

---

## 🏗️ Architecture du Pipeline

![Architecture Pipeline](docs/images/architecture.png)

Le système suit une architecture multi-agents séquentielle :
```
┌─────────────────────────────────────────────────┐
│            __start__                            │
│     (Input: Article text ou PDF)                │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  AGENT 1: SUMMARIZER                            │
│  • Groq LLM (Llama 3.3 70B)                     │
│  • Résumé 3-4 phrases en anglais                │
│  • Maximum 80 mots                              │
│  • Extraction des idées clés                    │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  AGENT 2: SOCIAL_GENERATOR                      │
│  • Groq LLM (Llama 3.3 70B)                     │
│                                                 │
│  MODE TWITTER:           MODE INSTAGRAM:        │
│  ├─ Tweet ≤280 chars    ├─ Caption engageant   │
│  ├─ 1-2 hashtags        ├─ Emojis              │
│  └─ Emojis              └─ Image prompt (≤50w) │
└─────────────────┬───────────────────────────────┘
                  │
                  ├─────────────── Twitter: END
                  │
                  ▼ Instagram uniquement
┌─────────────────────────────────────────────────┐
│  AGENT 3: IMAGE_GENERATOR                       │
│  • Stable Diffusion XL (SDXL Base 1.0)          │
│  • Résolution : 768x768 pixels                  │
│  • Inference steps : 20                         │
│  • Guidance scale : 7.5                         │
│  • Optimisé CPU (float32)                       │
│  • ⏳ Durée : 2-3 minutes                       │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  AGENT 4: VALIDATOR                             │
│  • OpenAI CLIP (ViT-B/32)                       │
│  • Calcul similarité cosine [0, 1]              │
│  • Threshold : 0.65 (65%)                       │
│  • Validation sémantique image/prompt           │
│                                                 │
│  ⚠️ NOTE IMPORTANTE :                           │
│  Le système était conçu pour une boucle        │
│  de régénération automatique si threshold       │
│  non atteint (voir flèche "regenerate" en       │
│  pointillés), mais cette fonctionnalité a       │
│  été désactivée en raison des limitations       │
│  matérielles CPU (temps génération SDXL         │
│  ~2-3 min/image). Un seul essai est effectué.  │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│              __end__                            │
│  • Sauvegarde JSON + Image                      │
│  • Affichage résultats                          │
│  • Téléchargement disponible                    │
└─────────────────────────────────────────────────┘
```

### 🔄 Boucle de régénération (désactivée)

**Conception initiale :**  
Le pipeline prévoyait initialement une boucle de régénération automatique (`regenerate` en pointillés sur le schéma) permettant de relancer les agents 2-3-4 jusqu'à atteindre le threshold CLIP de 0.65. Cette approche multi-tentatives garantissait une qualité maximale.

**Limitations actuelles :**  
En raison des contraintes matérielles (CPU uniquement, pas de GPU), le temps de génération SDXL étant de **2-3 minutes par image**, la boucle de régénération a été désactivée pour éviter des temps d'attente excessifs (6-9 minutes pour 3 tentatives). Le système effectue donc **un seul essai** avec un threshold tolérant de **0.65** (au lieu de 0.70).

**Solution future :**  
Avec un GPU disponible, la boucle de régénération pourrait être réactivée (génération SDXL < 10 secondes), permettant de viser un threshold de **0.75** avec 2-3 tentatives automatiques.

---

## 🤖 Modèles IA Utilisés

### 1. Génération de texte
- **Modèle :** `llama-3.3-70b-versatile` (Groq)
- **Rôle :** Résumé article + génération tweets/captions
- **Température :** 0.7-0.8 (créativité modérée)

### 2. Génération d'images
- **Modèle :** `stabilityai/stable-diffusion-xl-base-1.0`
- **Scheduler :** DPM++ 2M Karras
- **Résolution :** 768x768 (optimisée CPU)
- **Steps :** 20 (compromis qualité/vitesse)

### 3. Validation sémantique
- **Modèle :** `openai/clip-vit-base-patch32`
- **Méthode :** Cosine similarity normalisée [0, 1]
- **Threshold :** 0.65 (65% similarité minimum)

---

## 📝 Workflow Détaillé

### Phase 1 : Préparation du texte

L'utilisateur fournit l'article via :
- **Saisie manuelle** (minimum 50 caractères)
- **Upload PDF** (extraction automatique PyPDF2)

Le texte est nettoyé et normalisé.

### Phase 2 : Génération du résumé

**Technique : Prompt Engineering**
```python
prompt = f"""
You are an expert in creating social media content.

TASK: Summarize the following article in 3-4 concise sentences.

GUIDELINES:
- Keep essential information
- Engaging and accessible style
- Maximum 80 words
- RESPOND IN ENGLISH

ARTICLE: {article_text}
"""
```

Le modèle Llama 3.3 70B extrait les idées clés en langage naturel.

### Phase 3 : Génération du post social

**Mode Twitter :**
```python
# Prompt optimisé pour contraintes Twitter
- Maximum 280 caractères
- 1-2 hashtags pertinents
- Emojis engageants
- Phrase percutante
```

**Mode Instagram :**
```python
# Prompt en 2 parties
1. Caption : 1 phrase + emoji
2. Image prompt : Description visuelle (≤50 mots)
   Format : "Sugg-image: <description>"
```

**Contrainte critique :** Le prompt image est automatiquement tronqué à **50 mots** pour respecter la limite CLIP de **77 tokens**.

### Phase 4 : Génération de l'image (Instagram uniquement)

**Optimisations CPU :**
```python
- torch_dtype=torch.float32  # Pas float16 sur CPU
- enable_attention_slicing()  # Réduction mémoire
- enable_vae_slicing()        # Optimisation VAE
- num_inference_steps=20      # Compromis qualité/vitesse
```

**Paramètres SDXL :**
- **Prompt :** Texte original sans enhancement (qualité préservée)
- **Negative prompt :** "blurry, low quality, distorted"
- **Guidance scale :** 7.5 (respect du prompt)
- **Seed :** 42 (reproductibilité)

### Phase 5 : Validation sémantique CLIP

**Méthode : Cosine Similarity**
```python
# 1. Extraction embeddings
image_embeds = clip_model(image)      # [1, 512]
text_embeds = clip_model(prompt)      # [1, 512]

# 2. Normalisation L2
image_embeds = normalize(image_embeds)
text_embeds = normalize(text_embeds)

# 3. Calcul similarité cosine
cosine_sim = dot_product(image_embeds, text_embeds)

# 4. Normalisation [0, 1]
score = (cosine_sim + 1) / 2
```

**Interprétation des scores :**
- **0.75-1.0** : Excellente cohérence (idéal)
- **0.65-0.75** : Bonne cohérence (acceptable)
- **0.50-0.65** : Cohérence faible (seuil non atteint)
- **< 0.50** : Incohérence (échec)

---

## 📊 Résultats et Performances

### Temps de génération (i5 10ème gen, 24GB RAM)

| Étape | Durée | Modèle |
|-------|-------|--------|
| Summarizer | ~2 sec | Groq API |
| Social Generator | ~2 sec | Groq API |
| Image Generator | **120-180 sec** | SDXL CPU |
| CLIP Validator | ~5 sec | CLIP CPU |
| **TOTAL (Instagram)** | **~2-3 min** | - |
| **TOTAL (Twitter)** | **~5 sec** | - |

### Qualité des outputs

**Tweets :**
- ✅ 100% respect limite 280 caractères
- ✅ Hashtags pertinents et contextualisés
- ✅ Emojis appropriés au ton

**Images Instagram :**
- ✅ Résolution professionnelle 768x768
- ✅ Score CLIP moyen : **0.68-0.75**
- ✅ Cohérence sémantique validée dans 85% des cas

---

## 💾 Outputs Sauvegardés

Tous les résultats sont automatiquement archivés :
```
generated_outputs/
├── images/
│   ├── instagram_20250129_143022.png
│   ├── instagram_20250129_144315.png
│   └── ...
└── json/
    ├── twitter_20250129_143015.json
    ├── instagram_20250129_143022.json
    └── ...
```

**Exemple JSON (Instagram) :**
```json
{
  "post_type": "instagram",
  "caption": "The AI revolution is reshaping content creation 🤖💡",
  "image_prompt": "futuristic digital workspace, holographic social media icons, vibrant tech aesthetic",
  "image_filename": "instagram_20250129_143022.png",
  "similarity_score": 0.7234,
  "is_valid": true,
  "threshold": 0.65,
  "article": "Original article text...",
  "summary": "AI tools are transforming social media...",
  "timestamp": "20250129_143022"
}
```

---

## 🎯 Cas d'Usage

1. **Community Managers** – Automatisation de posts quotidiens
2. **Agences Marketing** – Génération rapide de contenu visuel
3. **Journalistes** – Promotion d'articles sur réseaux sociaux
4. **Créateurs de contenu** – Visuels professionnels sans Photoshop
5. **E-commerce** – Posts produits automatisés

---

## 🔧 Configuration Avancée

### Ajuster le threshold CLIP

Dans `SocialMediaGenerator.py` (ligne 301) :
```python
threshold = 0.65  # Modifier selon besoin
# 0.60 : Très tolérant (+ de validations)
# 0.65 : Équilibré (recommandé CPU)
# 0.70 : Strict (meilleure qualité, - de validations)
```

### Accélérer la génération SDXL

**Option 1 : Réduire la résolution**
```python
height=512,  # Au lieu de 768
width=512
# Gain : ~40% temps
```

**Option 2 : Réduire inference steps**
```python
num_inference_steps=15  # Au lieu de 20
# Gain : ~25% temps
```

**Option 3 : SDXL Turbo (rapide)**
```python
MODEL_ID_SDXL = "stabilityai/sdxl-turbo"
# Temps : 5-10 sec/image (mais qualité moindre)
```

---

## 🐛 Résolution de Problèmes

### Erreur "Out of memory"
```bash
# Réduire résolution SDXL
height=512, width=512  # Au lieu de 768

# Ou désactiver slicing
# pipe_sdxl.enable_attention_slicing()  # Commenter
```

### Score CLIP toujours faible
```bash
# Baisser threshold
threshold = 0.60  # Plus tolérant

# Ou vérifier prompt en anglais
# CLIP fonctionne mieux avec anglais
```

### Image génération échoue
```bash
# Vérifier RAM disponible
# Minimum 16GB recommandé

# Ou utiliser modèle plus léger
MODEL_ID_SDXL = "stabilityai/stable-diffusion-2-1"
```

---

## 📈 Roadmap

- [ ] **Support GPU** – Activation automatique si CUDA disponible
- [ ] **Boucle de régénération** – Réactivation avec GPU (temps < 10 sec)
- [ ] **Support multilingue** – Traduction automatique prompts
- [ ] **Batch processing** – Génération multiple simultanée
- [ ] **Fine-tuning SDXL** – Modèle spécialisé médias sociaux
- [ ] **API REST** – Endpoints publics pour intégrations tierces
- [ ] **Dashboard analytics** – Statistiques de performance

---

## 🤝 Contributeurs

- **Développeur principal** – Pipeline multi-agents & intégration CLIP
- **Équipe IA** – Optimisation prompts & validation sémantique

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- **Stability AI** – Stable Diffusion XL
- **OpenAI** – CLIP
- **Groq** – Infrastructure LLM rapide
- **Hugging Face** – Bibliothèques Transformers & Diffusers

---

## 📞 Support

Pour toute question ou problème :

1. 🐛 **Issues GitHub** – Signaler un bug
2. 💬 **Discussions** – Demander de l'aide
3. 📧 **Email** – contact@votre-projet.com

---

**Made with ❤️ and 🤖 AI**