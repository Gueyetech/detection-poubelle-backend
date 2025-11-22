# API de Détection de Poubelles - Backend FastAPI

Backend FastAPI pour la détection de poubelles pleines/vides avec YOLOv8.

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

### `GET /health`
Vérifier le statut de l'API et du modèle

### `POST /predict`
Upload une image et obtenir les détections

**Body:** `multipart/form-data`
- `file`: Image (JPG, PNG, etc.)

**Response:**
```json
{
  "success": true,
  "prediction_id": "uuid",
  "original_image": "/uploads/uuid.jpg",
  "annotated_image": "/results/uuid_annotated.jpg",
  "detections": [
    {
      "class": "poubelle_pleine",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "summary": {
    "total_detections": 1,
    "class_counts": {"poubelle_pleine": 1}
  }
}
```

### `POST /predict-batch`
Upload plusieurs images (max 10)

### `DELETE /cleanup/{prediction_id}`
Supprimer les fichiers associés à une prédiction

##  Structure

```
detection_poubelle_backend/
├── main.py              # Application FastAPI
├── model.py             # Gestion du modèle YOLO
├── requirements.txt     # Dépendances Python
├── .gitignore          
├── uploads/             # Images uploadées (créé auto)
└── results/             # Images annotées (créé auto)
```

##  Configuration

Le modèle YOLOv8 sera téléchargé automatiquement depuis GitHub au premier lancement.

##  Déploiement sur Render

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

##  Classes détectées

- `poubelle_pleine` : Poubelle pleine
- `poubelle_vide` : Poubelle vide
