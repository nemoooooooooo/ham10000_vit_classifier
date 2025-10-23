import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    """ViT-B/16 model with custom head for HAM10000"""
    def __init__(self, num_classes=7, model_name='vit_base_patch16_384', pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        self.feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True