# inference.py
import torch
from PIL import Image
from torchvision import transforms
from model import BirdClassifier
import io

class BirdPredictor:
    def __init__(self, model_path='model.pth'):
        checkpoint = torch.load(model_path, map_location='cpu')
        
        self.classes = checkpoint['classes']
        num_classes = len(self.classes)
        
        self.model = BirdClassifier(num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_from_bytes(self, image_bytes, top_k=3):
        """Prédiction à partir de bytes (pour FastAPI)"""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'species': self.classes[idx],
                'probability': float(prob.item())  # Convertir en float Python
            })
        
        return results
    
    def predict(self, image_path, top_k=3):
        """Prédiction à partir d'un chemin de fichier"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return self.predict_from_bytes(image_bytes, top_k)