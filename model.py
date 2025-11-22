from ultralytics import YOLO
from pathlib import Path
import urllib.request
import os
import cv2
from typing import Dict, List

# Configuration
MODEL_URL = "https://github.com/Gueyetech/train_detection_poubelle_plein_vide/raw/main/runs/detect/poubelle_pleine_vide7/weights/best.pt"
MODEL_PATH = Path("best.pt")

# Variable globale pour le modèle (lazy loading)
_model = None

def download_model():
    """Télécharge le modèle s'il n'existe pas"""
    if not MODEL_PATH.exists():
        print(f"Téléchargement du modèle depuis {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print(f"Modèle téléchargé avec succès vers {MODEL_PATH}")
        except Exception as e:
            print(f"Erreur lors du téléchargement: {e}")
            raise

def get_model():
    """Retourne le modèle YOLO (charge une seule fois)"""
    global _model
    
    if _model is None:
        download_model()
        print("Chargement du modèle YOLO...")
        _model = YOLO(str(MODEL_PATH))
        print("Modèle chargé avec succès !")
    
    return _model

def predict_image(image_path: str, output_dir: str, prediction_id: str) -> Dict:
    """
    Fait une prédiction sur une image
    
    Args:
        image_path: Chemin vers l'image
        output_dir: Dossier de sortie pour l'image annotée
        prediction_id: ID unique pour cette prédiction
    
    Returns:
        Dict avec les détections et le chemin de l'image annotée
    """
    model = get_model()
    
    # Faire la prédiction
    results = model.predict(image_path, save=False, conf=0.25)
    result = results[0]
    
    # Extraire les détections
    detections = []
    class_counts = {}
    
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        
        detections.append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "bbox": [round(x, 2) for x in bbox]
        })
        
        # Compter les classes
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Sauvegarder l'image annotée
    annotated_image = result.plot()
    output_path = Path(output_dir) / f"{prediction_id}_annotated.jpg"
    cv2.imwrite(str(output_path), annotated_image)
    
    return {
        "detections": detections,
        "summary": {
            "total_detections": len(detections),
            "class_counts": class_counts
        },
        "annotated_path": f"/results/{prediction_id}_annotated.jpg"
    }
