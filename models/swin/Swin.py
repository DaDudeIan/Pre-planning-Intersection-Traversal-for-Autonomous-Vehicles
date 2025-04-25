import timm
import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer

class Swin(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=5, image_size=400):
        super(Swin, self).__init__()
        self.image_size = image_size

        # Create Swin Transformer with custom image size
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            img_size=image_size,
            drop_rate=0.3
        )

        # Output feature dimensions from the last Swin stage
        feature_dim = self.backbone.feature_info[-1]['num_chs']

        # Simple decoder head
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        # Upsample to original image size
        self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)

    def forward(self, x):
        features = self.backbone(x)[-1]  # Use last feature map
        features = features.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        x = self.decoder(features)
        x = self.upsample(x)
        return x

