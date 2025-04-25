import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class ASPPv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPv3, self).__init__()
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.shape[2:]

        img_pool = self.image_pool(x)
        img_pool = F.interpolate(img_pool, size=size, mode='bilinear', align_corners=False)

        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)

        x = torch.cat([x1, x2, x3, x4, img_pool], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=5):
        super(DeepLabV3, self).__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        mobilenet = mobilenet_v3_large(weights=weights)
        self.backbone = mobilenet.features

        self.low_level_channels = 24  # from stage 2
        self.high_level_channels = 960  # from final conv block before classifier

        self.low_level_extractor = nn.Sequential(*self.backbone[:4])   # Up to index 3
        self.high_level_extractor = nn.Sequential(*self.backbone[4:])  # From index 4 to end

        self.aspp = ASPPv3(in_channels=self.high_level_channels, out_channels=256)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        size = x.shape[2:]

        low_level_feat = self.low_level_extractor(x)
        high_level_feat = self.high_level_extractor(low_level_feat)

        x = self.aspp(high_level_feat)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)

        low_level_feat = self.low_level_proj(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)

        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x