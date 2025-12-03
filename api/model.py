# model.py
import torch
import torch.nn as nn
import timm

class BirdClassifier(nn.Module):
    def __init__(self, num_classes, model_name='tf_efficientnetv2_s.in21k_ft_in1k'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        # Remplacer la dernière couche
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)