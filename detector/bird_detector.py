#!/usr/bin/env python3
"""
Bird Detection System using YOLOv8 and RTSP Camera
Détecte les oiseaux via une caméra RTSP et les identifie via l'API FastAPI
"""

import os
import sys
import cv2
import time
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BirdDetector:
    """Détecteur d'oiseaux via RTSP et YOLO"""
    
    def __init__(self):
        # Charger les variables d'environnement
        load_dotenv()
        
        # Configuration RTSP
        self.rtsp_url = os.getenv('RTSP_URL')
        if not self.rtsp_url:
            raise ValueError("RTSP_URL non défini dans .env")
        
        # Configuration API
        self.api_url = os.getenv('API_URL', 'http://localhost:8000')
        
        # Paramètres de détection
        self.confidence_threshold = float(os.getenv('YOLO_CONFIDENCE', '0.5'))
        self.cooldown = int(os.getenv('DETECTION_COOLDOWN', '5'))
        self.process_every_n_frames = int(os.getenv('PROCESS_EVERY_N_FRAMES', '10'))
        
        # État
        self.last_capture_time = 0
        self.frame_count = 0
        self.total_detections = 0
        
        # YOLO - classe 14 = bird dans COCO dataset
        self.bird_class_id = 14
        self.yolo = None
        
        # Dossiers de sortie
        self.captures_dir = Path(os.getenv('CAPTURES_DIR', '/captures'))
        self.captures_dir.mkdir(exist_ok=True)
        
        logger.info("BirdDetector initialisé")
        logger.info(f"RTSP URL: {self.rtsp_url[:30]}...")
        logger.info(f"API URL: {self.api_url}")
    
    def load_yolo(self):
        """Charge le modèle YOLO (lazy loading)"""
        if self.yolo is None:
            try:
                from ultralytics import YOLO
                logger.info("Chargement du modèle YOLOv11n...")
                self.yolo = YOLO('yolo11n.pt')
                logger.info("✅ Modèle YOLO chargé avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur lors du chargement de YOLO: {e}")
                raise
    
    def connect_rtsp(self):
        """Établit la connexion au flux RTSP"""
        logger.info("Connexion au flux RTSP...")

        # Configuration avec options FFMPEG pour améliorer la stabilité
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        # Configuration pour améliorer la stabilité et réduire les timeouts
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Options FFMPEG pour timeout plus court et reconnexion
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'

        if not cap.isOpened():
            raise ConnectionError(f"Impossible d'ouvrir le flux RTSP: {self.rtsp_url}")

        # Lire quelques frames pour initialiser
        for _ in range(5):
            ret, _ = cap.read()
            if not ret:
                logger.warning("Échec de lecture pendant l'initialisation")
                break

        logger.info("✅ Connexion RTSP établie")
        return cap
    
    def reconnect_rtsp(self, cap):
        """Reconnecte au flux RTSP en cas d'erreur"""
        logger.warning("Reconnexion au flux RTSP...")
        if cap:
            cap.release()
        time.sleep(2)
        return self.connect_rtsp()
    
    def detect_birds_in_frame(self, frame):
        """Détecte les oiseaux dans une frame avec YOLO"""
        try:
            results = self.yolo(frame, verbose=False, conf=self.confidence_threshold)
            
            birds = []
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Vérifier si c'est un oiseau
                    if class_id == self.bird_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        birds.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence
                        })
            
            return birds
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection: {e}")
            return []
    
    def capture_and_identify(self, frame, bird_info):
        """Capture l'oiseau détecté et l'identifie via l'API"""
        try:
            # Extraire la région de l'oiseau
            x1, y1, x2, y2 = bird_info['bbox']
            
            # Ajouter une marge de 10% autour de la détection
            height, width = frame.shape[:2]
            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(width, x2 + margin_x)
            y2 = min(height, y2 + margin_y)
            
            bird_crop = frame[y1:y2, x1:x2]
            
            # Vérifier que la crop n'est pas vide
            if bird_crop.size == 0:
                logger.warning("Image capturée vide, ignorée")
                return
            
            # Générer un nom de fichier temporaire
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_filename = f"temp_bird_{timestamp}.jpg"
            
            # Sauvegarder temporairement
            cv2.imwrite(temp_filename, bird_crop)
            
            logger.info(f"🐦 Oiseau détecté (confiance: {bird_info['confidence']:.2f}) - Identification en cours...")
            
            # Appeler l'API d'identification
            try:
                with open(temp_filename, 'rb') as f:
                    response = requests.post(
                        f"{self.api_url}/identify",
                        files={'input': f},
                        timeout=10
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data['predictions']
                    processing_time = data.get('processing_time_ms', 0)
                    
                    # Afficher les résultats
                    logger.info(f"✅ Identification réussie ({processing_time:.0f}ms):")
                    for i, pred in enumerate(predictions[:3], 1):
                        species = pred['species'].replace('_', ' ')
                        prob = pred['probability'] * 100
                        logger.info(f"   {i}. {species}: {prob:.1f}%")
                    
                    # Sauvegarder avec le nom de l'espèce
                    top_species = predictions[0]['species']
                    top_prob = predictions[0]['probability']
                    final_filename = f"{top_species}_{timestamp}_conf{top_prob:.2f}.jpg"
                    final_path = self.captures_dir / final_filename
                    
                    Path(temp_filename).rename(final_path)
                    logger.info(f"💾 Sauvegardé: {final_path}")
                    
                    self.total_detections += 1
                    
                else:
                    logger.error(f"❌ Erreur API: {response.status_code} - {response.text}")
                    Path(temp_filename).unlink(missing_ok=True)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Erreur de connexion à l'API: {e}")
                Path(temp_filename).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'identification: {e}")
                Path(temp_filename).unlink(missing_ok=True)
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la capture: {e}")
    
    def should_process_detection(self):
        """Vérifie si on doit traiter une nouvelle détection (cooldown)"""
        current_time = time.time()
        if current_time - self.last_capture_time >= self.cooldown:
            self.last_capture_time = current_time
            return True
        return False
    
    def start_detection(self):
        """Démarre la boucle de détection principale"""
        logger.info("=" * 60)
        logger.info("🚀 Démarrage du système de détection d'oiseaux")
        logger.info("=" * 60)
        
        # Charger YOLO
        self.load_yolo()
        
        # Connexion RTSP
        cap = self.connect_rtsp()
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        logger.info("👁️  Surveillance active... (Ctrl+C pour arrêter)")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    consecutive_errors += 1
                    logger.warning(f"Erreur de lecture frame ({consecutive_errors}/{max_consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        cap = self.reconnect_rtsp(cap)
                        consecutive_errors = 0
                    
                    time.sleep(0.1)
                    continue
                
                # Reset du compteur d'erreurs
                consecutive_errors = 0
                
                # Traiter seulement 1 frame sur N pour économiser CPU
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames != 0:
                    continue
                
                # Détection YOLO
                birds = self.detect_birds_in_frame(frame)
                
                # Si des oiseaux sont détectés
                if birds and self.should_process_detection():
                    # Prendre l'oiseau avec la meilleure confiance
                    best_bird = max(birds, key=lambda x: x['confidence'])
                    self.capture_and_identify(frame, best_bird)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Arrêt demandé par l'utilisateur")
        
        except Exception as e:
            logger.error(f"❌ Erreur fatale: {e}", exc_info=True)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info(f"📊 Statistiques: {self.total_detections} oiseaux identifiés")
            logger.info("👋 Arrêt du système de détection")


def main():
    """Point d'entrée principal"""
    try:
        detector = BirdDetector()
        detector.start_detection()
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()