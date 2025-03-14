import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Double convolution block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

# Down-sampling block: maxpool followed by double convolution
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

# Up-sampling block: upsample, then concatenate skip connection, then double convolution
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # Using bilinear upsampling followed by a 1x1 convolution to reduce the number of channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Alternatively, you can use a transposed convolution for upsampling
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust padding if necessary to handle odd input dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Final output layer: 1x1 convolution to map to the desired number of output channels
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

# U-Net model
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder pathway
        x1 = self.inc(x)      # [B, 64, 400, 400]
        x2 = self.down1(x1)   # [B, 128, 200, 200]
        x3 = self.down2(x2)   # [B, 256, 100, 100]
        x4 = self.down3(x3)   # [B, 512, 50, 50]
        x5 = self.down4(x4)   # [B, 1024, 25, 25]
        # Decoder pathway with skip connections
        x = self.up1(x5, x4)  # [B, 512, 50, 50]
        x = self.up2(x, x3)   # [B, 256, 100, 100]
        x = self.up3(x, x2)   # [B, 128, 200, 200]
        x = self.up4(x, x1)   # [B, 64, 400, 400]
        logits = self.outc(x) # [B, 1, 400, 400]
        return logits
    
def display_output(output, threshold=0.5, thresholded=False):
    """
    Displays binary masks from a model output that can handle multiple images in a batch.
    
    Parameters:
        output (torch.Tensor): The model's output tensor of shape [batch, 1, H, W].
        threshold (float): Threshold to convert probabilities to binary values.
    """
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(output)
    
    # Remove the channel dimension; resulting shape is [batch, H, W]
    masks = probs.squeeze(1).cpu().detach().numpy()
    
    batch_size = masks.shape[0]
    
    # Determine grid size (up to 4 columns)
    ncols = min(4, batch_size)
    nrows = (batch_size + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    
    # Ensure axes is 2D for uniform indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif ncols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for idx, mask in enumerate(masks):
        row = idx // ncols
        col = idx % ncols
        # Threshold the mask to get binary output and display it
        if thresholded:
            m = mask > threshold
        else:
            m = mask
        axes[row, col].imshow(m, cmap='gray')
        axes[row, col].set_title(f"Image {idx + 1}")
        axes[row, col].axis("off")
    
    # Hide any extra subplots if the grid is larger than the batch size
    for idx in range(batch_size, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    import cv2
    img = cv2.imread("satellite.png")
    img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    model = UNet(n_channels=3, n_classes=1)
    output = model(img_t)
    print("Output shape:", output.shape)  # Expected: torch.Size([1, 1, 400, 400])
    display_output(output)
    
    
