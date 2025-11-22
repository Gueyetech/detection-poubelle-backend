from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
from typing import List
import os
from datetime import datetime

from model import get_model, predict_image

app = FastAPI(title="Detection Poubelle API")

# CORS pour permettre React de communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier le domaine exact
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Créer les dossiers nécessaires
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Servir les fichiers statiques
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

@app.get("/")
async def root():
    return {
        "message": "API de Détection de Poubelles",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Vérifier que l'API et le modèle sont opérationnels"""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload une image et retourne les prédictions YOLO
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        # Générer un ID unique pour cette prédiction
        prediction_id = str(uuid.uuid4())
        
        # Sauvegarder l'image uploadée
        file_extension = Path(file.filename).suffix
        original_path = UPLOAD_DIR / f"{prediction_id}{file_extension}"
        
        with original_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Faire la prédiction
        result = predict_image(str(original_path), str(RESULTS_DIR), prediction_id)
        
        return JSONResponse(content={
            "success": True,
            "prediction_id": prediction_id,
            "original_image": f"/uploads/{prediction_id}{file_extension}",
            "annotated_image": result["annotated_path"],
            "detections": result["detections"],
            "summary": result["summary"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Upload plusieurs images et retourne les prédictions pour chacune
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images à la fois")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        
        try:
            prediction_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix
            original_path = UPLOAD_DIR / f"{prediction_id}{file_extension}"
            
            with original_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            result = predict_image(str(original_path), str(RESULTS_DIR), prediction_id)
            
            results.append({
                "success": True,
                "filename": file.filename,
                "prediction_id": prediction_id,
                "original_image": f"/uploads/{prediction_id}{file_extension}",
                "annotated_image": result["annotated_path"],
                "detections": result["detections"],
                "summary": result["summary"]
            })
            
        except Exception as e:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})

@app.delete("/cleanup/{prediction_id}")
async def cleanup(prediction_id: str):
    """
    Supprimer les fichiers associés à une prédiction
    """
    try:
        # Supprimer tous les fichiers avec cet ID
        for directory in [UPLOAD_DIR, RESULTS_DIR]:
            for file in directory.glob(f"{prediction_id}.*"):
                file.unlink()
        
        return {"success": True, "message": "Fichiers supprimés"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
