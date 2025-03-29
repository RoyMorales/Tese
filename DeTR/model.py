# DeTR Model

from typing import Optional
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from torchvision.models.resnet import resnet50


# NoPE First
# ToDO!
class PositionEncoding(nn.Module):
    """
    Fixed implementation of sinusoidal positional encoding that handles:
    - Both 3D (H,W) and 4D (B,H,W) input tensors
    - Proper dimension indexing
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: torch.Tensor):
        # Ensure proper dimensions
        if mask.dim() == 2:  # (H,W)
            mask = mask.unsqueeze(0)  # Add batch dim -> (1,H,W)
        elif mask.dim() == 3:  # (B,H,W)
            pass  # Already correct
        else:
            raise ValueError(f"Mask must be 2D or 3D, got {mask.dim()}D")
            
        not_mask = ~mask
        
        # Correct dimension indices for cumsum
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # Height dim (index 1)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # Width dim (index 2)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Remove batch dim if input didn't have one
        if mask.dim() == 2:
            pos = pos.squeeze(0)
            
        return pos 

# Done
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float =0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embeded dimension must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None):
        batch_size = query.size(0)
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Classical Self-Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(attn_output)


# Done
class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_model: int, hidden_dim, dropout: float =0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(F.gelu(self.linear1(x))))


# Done
class AddNorm(nn.Module):
    def __init__(self, embed_dim: int, dropout: float =0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sub_output: Tensor):
        return self.norm(x + self.dropout(sub_output))
    

# Done
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float =0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.add_norm1 = AddNorm(embed_dim, dropout)
        self.mlp = MultiLayerPerceptron(embed_dim, hidden_dim, dropout)
        self.add_norm2 = AddNorm(embed_dim, dropout) 

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src: Tensor, src_mask: Optional[Tensor] =None, src_key_padding_mask: Optional[Tensor] =None, pos: Optional[Tensor] =None):

        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, mask=src_mask)[0]

        #src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = self.add_norm1(src, src2)
        mlp_output = self.mlp(src)
        src = self.add_norm2(src, mlp_output)
        return src


# Done
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1, skip_first_layer_attn: bool =False):
        super().__init__()
        
        self.skip_first_layer_attn = skip_first_layer_attn

        if not skip_first_layer_attn:
            # Self-Attention
            self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
            self.add_norm1 = AddNorm(embed_dim, dropout)

        # Cross-Attention
        self.cross_attn = MultiLayerPerceptron(embed_dim, num_heads, dropout)
        self.add_norm2 = AddNorm(embed_dim, dropout)
        
        self.mlp = MultiLayerPerceptron(embed_dim, hidden_dim, dropout)
        self.add_norm3 = AddNorm(embed_dim, dropout)


    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor =None, memory_mask: Tensor =None, tgt_key_padding_mask: Optional[Tensor] =None, memory_key_padding_mask: Optional[Tensor] =None, pos: Optional[Tensor] =None, query_pos: Optional[Tensor] =None):

        if not skip_first_layer_attn:
            q = k = self.with_pos_embed(tgt, query_pos)
            self_attn_output = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            tgt = self.add_norm1(tgt, self_attn_output)

        tgt2 = self.cross_attn(query=self.with_pos_embed(tgt, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               values=memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.add_norm2(tgt, cross_attn_output)
        mlp_output = self.mlp(tgt)
        tgt = self.add_norm3(tgt, mlp_output)

        return tgt

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask: Optional[Tensor] =None, src_key_padding_mask: Optional[Tensor] =None, pos: Optional[Tensor] =None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        
    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_ket_padding_mask, pos, query_pos):

        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask= memory_mask, tgt_ket_padding_mask=tgt_ket_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)

        return output
    

# Done
class Transformer(nn.Module):
    def __init__(self, embed_dim: int =512, num_heads: int =8, num_encoder_layers: int =6, num_decoder_layers: int =6, hidden_dim: int =2048, dropout: float =0.1):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, hidden_dim, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        # Try Later
        #first_decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, hidden_dim, dropout, skip_first_layer_attn=True)

        self._reset_parameters()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    # Dont know what is this
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # Dont Know
    def forward(self, src: Tensor, mask: Tensor, query_embed: Tensor, pos_embed: Tensor):
        # Flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        # Initialize target (object queries)
        tgt = torch.zeros_like(query_embed)
        
        # Encoder (with positional encoding added at every layer)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        # Decoder (with query_pos and pos_embed added at every layer)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                         pos=pos_embed, query_pos=query_embed)
        
        return hs.transpose(1, 0), memory.permute(1, 2, 0).view(bs, c, h, w)  


class DeTR(nn.Module):
    def __init__(self, num_classes: int, num_queries: int =100, hidden_dim: int =256, num_heads: int =8, num_encoder_layers: int =6, num_decoder_layers: int =6):
        super().__init__()

        self.backbone = resnet50()
        del self.backbone.fc

        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = Transformer(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers
                )
        
        self.position_embedding = PositionEncoding(hidden_dim // 2)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)


    def forward(self, inputs: Tensor):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        src = self.conv(x)
        mask = torch.zeros(src.shape[-2:], dtype=torch.bool, device=src.device)
        pos = self.position_embedding(mask.to(src.device))

        hs, memory = self.transformer(src, mask, self.query_embed.weight, pos)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
    
        return {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}


class MLP(nn.Module):
    """Simple multi-layer perceptron for bbox prediction"""
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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



if __name__ == "__main__":
    #MultiHeadAttention
    print("MultiHeadAttention: ")
    test_inputMHA = torch.randn(2, 100, 256)
    mha = MultiHeadAttention(embed_dim=256, num_heads=8, dropout=0.1)
    outputMHA = mha(test_inputMHA[0], test_inputMHA[1], test_inputMHA[1])
    print("Output: ", outputMHA.shape)
    print("")

    #MLP
    print("MultiLayerPerceptron: ")
    test_inputMLP = torch.randn(2, 100)
    mlp = MultiLayerPerceptron(d_model=100, hidden_dim=400, dropout=0.1)
    outputMLP = mlp(test_inputMLP)
    print("Output: ", outputMLP.shape)
    print("")


    print("DeTR: ")









