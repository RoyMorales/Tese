# Implementation of DeTR

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

import math
import copy
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, pos_embed=None):
        batch_size = query.size(0)
        
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Add positional embeddings
        if pos_embed is not None:
            q = q + pos_embed
            k = k + pos_embed
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply masks
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
            
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2, _ = self.self_attn(src, src, src, 
                                key_padding_mask=src_key_padding_mask,
                                attn_mask=src_mask,
                                pos_embed=pos)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        q = tgt + query_pos if query_pos is not None else tgt
        k = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.self_attn(q, k, tgt,
                                key_padding_mask=tgt_key_padding_mask,
                                attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2, _ = self.multihead_attn(
            tgt + (query_pos if query_pos is not None else 0),
            memory + (pos if pos is not None else 0),
            memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=memory_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Encoder layers
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])
        
        # Decoder layers
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        
        # Encoder
        memory = src
        for layer in self.encoder:
            memory = layer(memory, src_mask=src_mask,
                         src_key_padding_mask=src_key_padding_mask,
                         pos=pos)
        
        # Decoder
        output = tgt
        for layer in self.decoder:
            output = layer(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         pos=pos, query_pos=query_pos)
        
        return output, memory

class PositionEmbeddingSine(nn.Module):
    """Fixed implementation that handles proper tensor dimensions"""
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale or 2 * math.pi
        
    def forward(self, mask):
        # Ensure mask is 4D: [batch_size, height, width]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add batch dimension if missing
        elif mask.dim() == 3:
            pass  # Already correct [batch_size, height, width]
        else:
            raise ValueError(f"Mask must be 2D or 3D, got {mask.dim()}D")
            
        not_mask = ~mask
        
        # Compute positional embeddings
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # Height dimension
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # Width dimension
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), 
                           pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), 
                           pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Remove batch dimension if input was 2D
        if mask.dim() == 2:
            pos = pos.squeeze(0)
            
        return pos

class DeTR(nn.Module):
    def __init__(self, num_classes, num_queries=100, hidden_dim=256,
                 nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        # Backbone
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc
        
        # Transformer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        
        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)

    def forward(self, inputs):
        # Backbone
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
    
        # Transformer
        src = self.conv(x)
        batch_size, _, height, width = src.shape
    
        # Create proper mask [batch_size, height, width]
        mask = torch.zeros((batch_size, height, width), 
                          dtype=torch.bool, 
                          device=src.device)
    
        pos = self.position_embedding(mask)
    
        # Prepare inputs
        src = src.flatten(2).permute(2, 0, 1)  # [hw, batch, channels]
        pos = pos.flatten(2).permute(2, 0, 1)   # [hw, batch, channels]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = torch.zeros_like(query_embed)
    
        # Transformer forward pass
        hs, memory = self.transformer(
            src=src,
            tgt=tgt,
            pos=pos,
            query_pos=query_embed
        )
    
        # Outputs
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
    
        return {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
