# api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

from inference import BirdPredictor

app = FastAPI(
    title="Bird Species Classifier API",
    description="API pour identifier les espèces d'oiseaux de Nouvelle-Aquitaine",
    version="1.0.0"
)

# CORS (si vous avez un frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle de réponse
class PredictionResult(BaseModel):
    species: str
    probability: float

class IdentifyResponse(BaseModel):
    predictions: List[PredictionResult]
    processing_time_ms: float

# Charger le modèle au démarrage (une seule fois)
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    model_path = Path("/model/model.pth")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
    
    print(f"Chargement du modèle depuis {model_path}...")
    predictor = BirdPredictor(model_path=str(model_path))
    print("Modèle chargé avec succès !")
    print(f"Nombre de classes : {len(predictor.classes)}")

@app.get("/")
def read_root():
    return {
        "message": "Bird Species Classifier API",
        "endpoints": {
            "/identify": "POST - Identifier une espèce d'oiseau",
            "/species": "GET - Liste des espèces connues",
            "/health": "GET - Status de l'API"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "num_species": len(predictor.classes) if predictor else 0
    }

@app.get("/species")
def list_species():
    """Liste toutes les espèces d'oiseaux reconnues par le modèle"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "species": predictor.classes,
        "count": len(predictor.classes)
    }

@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    input: UploadFile = File(..., description="Image d'oiseau à identifier")
):
    """
    Identifier l'espèce d'un oiseau à partir d'une image
    
    - **input**: Fichier image (JPG, PNG, etc.)
    
    Retourne les 3 espèces les plus probables avec leurs scores de confiance
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    # Vérifier le type de fichier
    if not input.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Le fichier doit être une image. Type reçu : {input.content_type}"
        )
    
    try:
        # Lire l'image
        import time
        start_time = time.time()
        
        image_bytes = await input.read()
        
        # Prédiction
        predictions = predictor.predict_from_bytes(image_bytes, top_k=3)
        
        processing_time = (time.time() - start_time) * 1000  # en ms
        
        return {
            "predictions": predictions,
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )

@app.post("/identify/detailed")
async def identify_detailed(
    input: UploadFile = File(...),
    top_k: int = 5
):
    """
    Version détaillée avec plus d'informations
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        import time
        start_time = time.time()
        
        image_bytes = await input.read()
        predictions = predictor.predict_from_bytes(image_bytes, top_k=top_k)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "filename": input.filename,
            "content_type": input.content_type,
            "predictions": predictions,
            "processing_time_ms": round(processing_time, 2),
            "top_prediction": predictions[0] if predictions else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))