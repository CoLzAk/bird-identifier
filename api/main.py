# api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from inference import BirdPredictor
from email_utils import send_bird_detection_email
from species_utils import load_species_with_thumbnails
import os

load_dotenv()

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
    """Liste toutes les espèces d'oiseaux reconnues par le modèle avec leurs noms communs et images"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Load species data with base64 thumbnails
    species_data = load_species_with_thumbnails()

    return {
        "species": species_data,
        "count": len(species_data)
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

        if (predictions[0]['probability'] < float(os.getenv('MODEL_CONFIDENCE', '0.6'))):
            return {
                "predictions": [],
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }

        processing_time = (time.time() - start_time) * 1000  # en ms

        # Send email notification (non-blocking, errors are logged but don't fail the request)
        try:
            await send_bird_detection_email(
                image_bytes=image_bytes,
                predictions=predictions,
                filename=input.filename or "bird_detection.jpg"
            )
        except Exception as email_error:
            print(f"⚠️ Email notification failed: {str(email_error)}")

        return {
            "predictions": predictions,
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )