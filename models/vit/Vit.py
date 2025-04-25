import torch
from torch import nn
import torch.nn.functional as F 
from torchvision.models.vision_transformer import VisionTransformer, vit_b_16

class ViT(nn.Module):
    def __init__(self, image_size=400, patch_size=16, num_classes=5, hidden_dim=768, 
                 num_layers=12, num_heads=12, mlp_dim=3072, dropout_rate=0.2, attention_dropout_rate=0.2):
        super(ViT, self).__init__()

        # Create a VisionTransformer from torchvision directly
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            attention_dropout=attention_dropout_rate,
            num_classes=0  # no classification head
        )

        # Compute grid size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.encoder_dropout = nn.Dropout(dropout_rate)
        
        # Fix positional embedding - debug the shapes to verify
        orig_pos_embed = self.vit.encoder.pos_embedding  # Shape: [1, N+1, hidden_dim]
        
        # Check if we need to interpolate or resize the positional embedding
        if orig_pos_embed.shape[1] != self.num_patches + 1:
            print(f"Resizing position embedding from {orig_pos_embed.shape} to [1, {self.num_patches + 1}, {hidden_dim}]")
            
            # Extract class token and patch embeddings
            cls_pos_embed = orig_pos_embed[:, 0:1, :]
            patch_pos_embed = orig_pos_embed[:, 1:, :]
            
            # Get spatial dimensions
            orig_grid_size = int(patch_pos_embed.shape[1] ** 0.5)
            new_grid_size = self.grid_size
            
            if orig_grid_size != new_grid_size:
                print(f"Resizing patch position embedding from {orig_grid_size}x{orig_grid_size} to {new_grid_size}x{new_grid_size}")
                # Resize patch position embedding
                patch_pos_embed = patch_pos_embed.reshape(1, orig_grid_size, orig_grid_size, hidden_dim)
                patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, hidden_dim, H, W]
                
                # Perform interpolation
                patch_pos_embed = F.interpolate(
                    patch_pos_embed,
                    size=(new_grid_size, new_grid_size),
                    mode='bicubic',
                    align_corners=False
                )
                
                # Reshape back
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # [1, H, W, hidden_dim]
                patch_pos_embed = patch_pos_embed.flatten(1, 2)  # [1, H*W, hidden_dim]
                
                # Combine with class token
                new_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
                
                # Update the positional embedding
                self.vit.encoder.pos_embedding = nn.Parameter(new_pos_embed)
                
                print(f"New positional embedding shape: {self.vit.encoder.pos_embedding.shape}")

        # Segmentation head
        self.pre_seg_dropout = nn.Dropout2d(dropout_rate)
        self.segmentation_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # Get the input shape
        B = x.shape[0]
        
        # Process the input, but don't use self.vit._process_input directly
        # because we need to handle the class token explicitly
        
        # 1. Extract patches
        x = self.vit.conv_proj(x)  # [B, hidden_dim, grid_size, grid_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        
        # 2. Add class token
        cls_token = self.vit.class_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches+1, hidden_dim]
        
        # 3. Add position embeddings
        x = x + self.vit.encoder.pos_embedding
        
        # Now run through encoder
        x = self.vit.encoder(x)  # [B, num_patches+1, hidden_dim]
        
        x = self.encoder_dropout(x)
        
        # Drop class token
        x = x[:, 1:, :]  # [B, num_patches, hidden_dim]
        
        # Reshape to spatial dimensions
        H = W = self.grid_size
        x = x.reshape(B, H, W, -1)  # [B, grid_size, grid_size, hidden_dim]
        x = x.permute(0, 3, 1, 2)   # [B, hidden_dim, grid_size, grid_size]
        
        x = self.pre_seg_dropout(x)
        # Segmentation head
        x = self.segmentation_head(x)
        
        # Upsample to original image size
        x = F.interpolate(x, size=(400, 400), 
                        mode='bilinear', align_corners=False)
        
        return x