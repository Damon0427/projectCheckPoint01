import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class TemporalAttentionPooling(nn.Module):
    def __init__(self, embed_dim, attn_dim=256, dropout=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, D)
        scores = self.attn(x)              # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)       # (B, D)
        return pooled, weights


class SRMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        kernels = torch.tensor([
            [[[ 0,-1, 0],[-1, 4,-1],[ 0,-1, 0]]],
            [[[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]],
            [[[ 1, 2, 1],[ 0, 0, 0],[-1,-2,-1]]],
        ], dtype=torch.float32)  # (3, 1, 3, 3)
        self.conv = nn.Conv2d(1, 3, 3, padding=1, bias=False)
        self.conv.weight = nn.Parameter(kernels, requires_grad=False)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)   # (N, 1, H, W)
        out  = self.conv(gray)               # (N, 3, H, W)
        return torch.tanh(out / 4.0)
    


class DualChannelDeepfakeDetector(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", freeze_vit=True, dropout=0.3):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.embed_dim = self.vit.config.hidden_size  # 768
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.temporal_pool_rgb = TemporalAttentionPooling(
            embed_dim=self.embed_dim, attn_dim=256, dropout=dropout
        )

        self.srm = SRMLayer()  

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet = nn.Sequential(*list(backbone.children())[:-1])

        # frozen all layer first
        for p in self.efficientnet.parameters():
            p.requires_grad = False

        # unfreeze the last layer for learning SRM feature
        for p in self.efficientnet[-1].parameters():   
            p.requires_grad = True

        self.freq_project = nn.Sequential(
            nn.Flatten(),          
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.temporal_pool_freq = TemporalAttentionPooling(
            embed_dim=256, attn_dim=128, dropout=dropout
        )

        
        # feat1: 768, feat2: 256 → concat → 1024
        self.fusion = nn.Sequential(
            nn.Linear(768 + 256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def encode_rgb(self, x):
        B, T, C, H, W = x.shape

        # flatten all of the frame with number of batch , and processe to ViT
        x_flat = x.view(B * T, C, H, W)

        # Extract CLS token from ViT output for each frame  (B*T, 768)
        cls = self.vit(pixel_values=x_flat).last_hidden_state[:, 0] 

        # Restore temporal structure for temporal pooling layer:
        cls = cls.view(B, T, self.embed_dim)
        feat, attn_w = self.temporal_pool_rgb(cls)   # (B, 768)
        return feat, attn_w

    def encode_freq(self, x):
        """x: (B, T, C, H, W)"""
        B, T, C, H, W = x.shape
        # flatten all of the frame with number of batch , and processe to ViT

        x_flat = x.view(B * T, C, H, W)

        # Apply SRM filter / frequency enhancement on each frame with size (B*T, 3, H, W)
        srm_out = self.srm(x_flat)                   
        # Extract frame-level frequency features using EfficientNet with size (B*T, 1280, 1, 1)
        feat = self.efficientnet(srm_out)             
        feat = self.freq_project(feat)               
        # Restore temporal structure for temporal pooling layer:
        feat = feat.view(B, T, 256)
        feat, attn_w = self.temporal_pool_freq(feat) 
        return feat, attn_w

    def forward(self, x):
        feat_rgb,  attn_rgb  = self.encode_rgb(x)   
        feat_freq, attn_freq = self.encode_freq(x) 

        # concated feature rgb and feature frequency( 768 + 256)
        concated  = torch.cat([feat_rgb, feat_freq], dim=-1)  
        # fused the concated feature 
        fused  = self.fusion(concated)
        # model classified the concated feature                          
        logits = self.classifier(fused).squeeze(1)          

        return logits, fused, attn_rgb