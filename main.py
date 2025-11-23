from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
from typing import List
import os
import cv2
import numpy as np
import base64

from model import get_model, predict_image as predict_image_model

app = FastAPI(
	title="Trash Detection API",
	version="1.0.0",
	description="API de détection de poubelles pleines ou vides utilisant YOLOv8",
	contact={
		"name": "Gueye Tech",
		"url": "https://github.com/Gueyetech",
	},
	license_info={
		"name": "MIT",
	}
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Créer les dossiers nécessaires
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Monter les dossiers statiques
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

@app.get("/", tags=["Info"], summary="Page d'accueil")
async def root():

	return {
		"message": "Trash Detection API",
		"version": "1.0.0",
		"documentation": {
			"swagger": "/docs",
			"openapi_json": "/openapi.json"
		},
		"endpoints": {
			"info": ["/", "/api/health", "/api/info"],
			"prediction": ["/api/predict/image", "/api/predict/video"],
			"model": ["/api/model/download", "/api/model/info"],
		}
	}

@app.get("/api/health", tags=["Info"], summary="Vérification de l'état de santé")
async def health_check():
	"""
	Vérifie l'état de santé de l'API et du modèle YOLO.
	
	Retourne:
	- **status**: 'healthy' si le modèle est chargé, 'unhealthy' sinon
	- **model_loaded**: True si le modèle est chargé avec succès
	- **error**: Message d'erreur en cas de problème
	"""
	try:
		model = get_model()
		return {
			"status": "healthy",
			"model_loaded": True
		}
	except Exception as e:
		return {
			"status": "unhealthy",
			"model_loaded": False,
			"error": str(e)
		}

@app.post("/api/predict/image", tags=["Prédiction"], summary="Détection sur une image")
async def predict_image_endpoint(file: UploadFile = File(..., description="Fichier image (JPG, PNG, etc.)")):
	"""
	Effectue une détection de poubelles sur une image.
	
	Paramètres:
	- **file**: Fichier image à analyser (formats supportés: JPG, PNG, BMP, etc.)
	
	Retourne:
	- **success**: Indique si la prédiction a réussi
	- **prediction_id**: Identifiant unique de la prédiction
	- **uploaded_file**: URL de l'image originale uploadée
	- **annotated_image**: URL de l'image annotée avec les détections
	- **detections**: Liste des objets détectés avec leurs détails
	  - class: Nom de la classe (poubelle_pleine ou poubelle_vide)
	  - confidence: Score de confiance (0-1)
	  - bbox: Boîte englobante [x1, y1, x2, y2]
	- **summary**: Résumé des détections
	  - total_detections: Nombre total d'objets détectés
	  - class_counts: Nombre de détections par classe
	
	Exemple de réponse:
	```json
	{
	  "success": true,
	  "prediction_id": "abc-123",
	  "uploaded_file": "/uploads/abc-123_image.jpg",
	  "annotated_image": "/results/abc-123_annotated.jpg",
	  "detections": [
	    {
	      "class": "poubelle_pleine",
	      "confidence": 0.95,
	      "bbox": [100, 150, 300, 400]
	    }
	  ],
	  "summary": {
	    "total_detections": 1,
	    "class_counts": {"poubelle_pleine": 1}
	  }
	}
	```
	"""
	try:
		if not file.content_type.startswith("image/"):
			raise HTTPException(status_code=400, detail="Fichier doit être une image")
        
		# Sauvegarder le fichier uploadé
		prediction_id = str(uuid.uuid4())
		file_path = UPLOAD_DIR / f"{prediction_id}_{file.filename}"
		
		with open(file_path, "wb") as buffer:
			shutil.copyfileobj(file.file, buffer)
        
		# Faire la prédiction
		result = predict_image_model(str(file_path), str(RESULTS_DIR), prediction_id)
        
		return JSONResponse({
			"success": True,
			"prediction_id": prediction_id,
			"uploaded_file": f"/uploads/{prediction_id}_{file.filename}",
			"annotated_image": result["annotated_path"],
			"detections": result["detections"],
			"summary": result["summary"]
		})
	except HTTPException:
		raise
	except Exception as e:
		# Nettoyer les fichiers en cas d'erreur
		if 'file_path' in locals() and file_path.exists():
			file_path.unlink()
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/video", tags=["Prédiction"], summary="Détection sur une vidéo")
async def predict_video(file: UploadFile = File(..., description="Fichier vidéo (MP4, AVI, etc.)")):
	"""
	Effectue une détection de poubelles sur toutes les frames d'une vidéo.
	
	Paramètres:
	- **file**: Fichier vidéo à analyser
	
	Retourne:
	- **success**: Indique si le traitement a réussi
	- **video**: Vidéo annotée encodée en base64
	- **frames_processed**: Nombre de frames traitées
	- **total_detections**: Nombre total de détections dans toute la vidéo
	- **average_detections_per_frame**: Moyenne de détections par frame
	- **detection_stats**: Statistiques par classe
	- **video_info**: Informations sur la vidéo (fps, dimensions, etc.)
	
	⚠️ **Attention**: Cette opération peut être longue pour les vidéos volumineuses.
	La vidéo de sortie est encodée en base64, ce qui peut être limité pour les fichiers très larges.
	"""
	try:
		if not file.content_type.startswith("video/"):
			raise HTTPException(status_code=400, detail="Fichier doit être une vidéo")
        
		# Obtenir le modèle
		model = get_model()
        
		temp_dir = Path("temp_videos")
		temp_dir.mkdir(exist_ok=True)
        
		input_path = temp_dir / file.filename
		with open(input_path, "wb") as buffer:
			shutil.copyfileobj(file.file, buffer)
        
		cap = cv2.VideoCapture(str(input_path))
		if not cap.isOpened():
			raise HTTPException(status_code=400, detail="Impossible de lire la vidéo")
        
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
		output_path = temp_dir / f"output_{file.filename}"
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
		frame_count = 0
		total_detections = 0
		detection_stats = {}
        
		while True:
			ret, frame = cap.read()
			if not ret:
				break
            
			results = model(frame, conf=0.25, verbose=False)
			annotated_frame = results[0].plot()
			out.write(annotated_frame)
            
			boxes = results[0].boxes
			frame_detections = len(boxes)
			total_detections += frame_detections
            
			for box in boxes:
				class_name = results[0].names[int(box.cls[0])]
				detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
            
			frame_count += 1
        
		cap.release()
		out.release()
        
		with open(output_path, "rb") as video_file:
			video_bytes = video_file.read()
			video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        
		input_path.unlink()
		output_path.unlink()
        
		return JSONResponse({
			"success": True,
			"video": f"data:video/mp4;base64,{video_base64}",
			"frames_processed": frame_count,
			"total_detections": total_detections,
			"average_detections_per_frame": round(total_detections / frame_count, 2) if frame_count > 0 else 0,
			"detection_stats": detection_stats,
			"video_info": {
				"fps": fps,
				"width": width,
				"height": height,
				"total_frames": total_frames
			}
		})
	except HTTPException:
		raise
	except Exception as e:
		if 'input_path' in locals() and input_path.exists():
			input_path.unlink()
		if 'output_path' in locals() and output_path.exists():
			output_path.unlink()
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/info", tags=["Info"], summary="Informations sur le modèle")
async def get_info():
	"""
	Retourne les informations sur le modèle YOLO chargé.
	
	Retourne:
	- **model_type**: Type de modèle utilisé
	- **classes**: Dictionnaire des classes détectables (ID -> nom)
	- **num_classes**: Nombre de classes
	- **endpoints**: Liste des endpoints disponibles
	
	En cas d'erreur:
	- **error**: Message d'erreur
	- **detail**: Détails de l'erreur
	"""
	try:
		model = get_model()
		return {
			"model_type": "YOLOv8",
			"classes": model.names,
			"num_classes": len(model.names),
			"documentation": {
				"swagger_ui": "/docs",
				"redoc": "/redoc",
				"openapi_schema": "/openapi.json"
			},
			"endpoints": {
				"info": ["/", "/api/health", "/api/info"],
				"prediction": ["/api/predict/image", "/api/predict/video"],
				"model": ["/api/model/download", "/api/model/info"],
				"management": ["/api/cleanup/{prediction_id}"]
			}
		}
	except Exception as e:
		return {
			"error": "Modèle non chargé",
			"detail": str(e)
		}

@app.delete("/api/cleanup/{prediction_id}", tags=["Gestion"], summary="Nettoyage des fichiers")
async def cleanup_prediction(prediction_id: str):
	"""
	Supprime tous les fichiers associés à une prédiction.
	
	Paramètres:
	- **prediction_id**: Identifiant unique de la prédiction à nettoyer
	
	Retourne:
	- **success**: Indique si le nettoyage a réussi
	- **deleted_files**: Liste des noms de fichiers supprimés
	
	Utilisé pour libérer l'espace disque après avoir traité les résultats.
	"""
	try:
		deleted_files = []
		
		# Chercher et supprimer les fichiers dans uploads
		for file in UPLOAD_DIR.glob(f"{prediction_id}_*"):
			file.unlink()
			deleted_files.append(str(file.name))
		
		# Chercher et supprimer les fichiers dans results
		for file in RESULTS_DIR.glob(f"{prediction_id}_*"):
			file.unlink()
			deleted_files.append(str(file.name))
		
		return {
			"success": True,
			"deleted_files": deleted_files
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/download", tags=["Modèle"], summary="Télécharger le modèle YOLO")
async def download_model_endpoint():
	"""
	Télécharge le fichier du modèle YOLO (.pt).
	
	Retourne le fichier best.pt utilisé pour les détections.
	Utile pour:
	- Sauvegarder une copie du modèle
	- Utiliser le modèle localement
	- Partager le modèle avec d'autres services
	
	Le fichier est téléchargé automatiquement au premier démarrage de l'API si absent.
	"""
	from model import MODEL_PATH, download_model as download_model_func
	
	try:
		# Télécharger le modèle s'il n'existe pas
		if not MODEL_PATH.exists():
			download_model_func()
		
		if not MODEL_PATH.exists():
			raise HTTPException(status_code=404, detail="Modèle non disponible")
		
		return FileResponse(
			path=str(MODEL_PATH),
			filename="best.pt",
			media_type="application/octet-stream"
		)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/info", tags=["Modèle"], summary="Informations sur le fichier modèle")
async def model_file_info():
	"""
	Retourne les informations sur le fichier du modèle.
	
	Retourne:
	- **exists**: Si le fichier modèle existe localement
	- **path**: Chemin du fichier modèle
	- **size_mb**: Taille du fichier en Mo (si disponible)
	- **download_url**: URL de téléchargement du modèle depuis GitHub
	"""
	from model import MODEL_PATH, MODEL_URL
	
	exists = MODEL_PATH.exists()
	size_mb = None
	
	if exists:
		size_bytes = MODEL_PATH.stat().st_size
		size_mb = round(size_bytes / (1024 * 1024), 2)
	
	return {
		"exists": exists,
		"path": str(MODEL_PATH),
		"size_mb": size_mb,
		"download_url": MODEL_URL
	}

if __name__ == "__main__":
	import uvicorn
	port = int(os.environ.get("PORT", 8000))
	uvicorn.run(app, host="0.0.0.0", port=port)