# Vision Transformer Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def self_attention(q, k, head_dim: int):
    return (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float =0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must to be divisible by the number of Heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = self_attention(q, k, self.head_dim)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        out = (attention @ v).transpose(1, 2).reshape(B, N, C)
        return self.fc_out(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_exp: float =4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_dim_exp = int(embed_dim * mlp_exp)

        self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim_exp),
                nn.GELU(), #Or Other Activation Function -> Do not use ReLU because of bad performance
                nn.Dropout(dropout),
                nn.Linear(mlp_dim_exp, embed_dim),
                nn.Dropout(dropout)
                )

    def forward(self, x):
        # Residual Connections
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, num_classes: int, embed_dim: int, depth: int, num_heads: int, mlp_exp: float =4.0, dropout: float =0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
                TransformerEncoderBlock(embed_dim, num_heads, mlp_exp, dropout)
                for _ in range(depth)
                ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Position Encoding
        # Addiction Method done only before the transformers sequence
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0]) # Extract CLS Token
        return self.head(x)


if __name__ == "__main__":
    model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_classes=10, embed_dim=768, depth=12, num_heads=8)
    input_test = torch.randn(1, 3, 224, 224)
    output = model(input_test)
    print("Output Shape: ", output.shape)
    






        














