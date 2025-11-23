#  Application de Détection de Poubelles - Streamlit + FastAPI

Application complète de détection de poubelles pleines/vides utilisant YOLOv8, avec interface web Streamlit et API FastAPI.

##  Installation

```bash
# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement virtuel
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

##  Lancement

### Interface Streamlit (Recommandée)

```bash
streamlit run streamlit_app.py
```

L'application web sera accessible à : http://localhost:8501

**Fonctionnalités :**
-  **Upload d'images** : Téléchargez des images depuis votre ordinateur
-  **Appareil photo/Webcam** : Prenez des photos en direct (fonctionne sur mobile et ordinateur)
-  **Analyse vidéo** : Traitez des vidéos complètes
-  **Statistiques détaillées** : Visualisez les détections et leurs confidences

### API FastAPI

```bash
# Mode développement
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Mode production
python main.py
```

L'API sera accessible à : http://localhost:8000

Documentation interactive : http://localhost:8000/docs

##  Endpoints

### `GET /`
Informations sur l'API


##  Structure

```
detection_poubelle_backend/
├── streamlit_app.py     # Application web Streamlit (Interface utilisateur)
├── main.py              # API FastAPI
├── model.py             # Gestion du modèle YOLO
├── best.pt              # Modèle YOLOv8 entraîné
├── requirements.txt     # Dépendances Python
├── Dockerfile           # Configuration Docker          
├── uploads/             # Images uploadées (créé auto)
├── results/             # Images annotées (créé auto)
└── temp_videos/         # Vidéos temporaires (créé auto)
```

##  Utilisation sur Mobile

L'application Streamlit fonctionne parfaitement sur mobile :

1. **Déployez l'application** sur Streamlit Cloud, Render, ou Heroku
2. **Ouvrez l'URL** de votre application sur votre téléphone
3. **Accédez à l'onglet " Appareil Photo"**
4. **Autorisez l'accès** à l'appareil photo de votre téléphone
5. **Prenez une photo** et l'analyse se fera automatiquement

### Astuce Mobile
- Utilisez le mode portrait pour une meilleure expérience
- L'appareil photo s'active automatiquement sur mobile
- Les résultats s'affichent immédiatement après la capture

##  Configuration

Le modèle YOLOv8 (`best.pt`) doit être présent dans le répertoire ou sera téléchargé automatiquement depuis GitHub au premier lancement.

##  Déploiement

### Déploiement Streamlit Cloud (Recommandé pour l'interface)

1. **Poussez votre code** sur GitHub
2. **Connectez-vous** à [Streamlit Cloud](https://streamlit.io/cloud)
3. **Déployez** en sélectionnant votre repository
4. **Fichier principal** : `streamlit_app.py`

### Déploiement sur Render (API FastAPI)

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Déploiement Docker

```bash
docker build -t detection-poubelle .

docker run -p 8501:8501 detection-poubelle
```

##  Classes détectées

- `poubelle_pleine`  : Poubelle pleine
- `poubelle_vide`  : Poubelle vide

## Technologies utilisées

- **YOLOv8s* : Modèle de détection d'objets
- **Streamlit** : Interface web interactive
- **FastAPI** : API REST haute performance
- **OpenCV** : Traitement d'images et vidéos
- **Pillow** : Manipulation d'images
- **Ultralytics** : Framework YOLO

##  Notes

- Les images et vidéos uploadées sont sauvegardées temporairement
- Les résultats annotés sont générés automatiquement
- Le modèle fonctionne avec une confiance minimale de 25% par défaut
- Compatible mobile et desktop

##  Développeur

Développé par **Gueye Tech**

[GitHub](https://github.com/Gueyetech)
