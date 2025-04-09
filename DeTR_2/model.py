#DeTR implementation 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.nn.modules.transformer import MultiheadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.nn.modules.transformer import MultiheadAttention

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self._reset_parameters()
        
        self.d_model = d_model
        self.nhead = nhead
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        # Encode the source (image features)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        # Decode to get predictions
        hs = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask,
                          pos=pos, query_pos=query_pos)
        return hs


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        # Self attention with positional encoding
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        # Self attention
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if pos is None else memory + pos
        tgt2 = self.multihead_attn(q, k, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
        return output


def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])


class DETR(nn.Module):
    def __init__(self, num_classes, num_queries, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        # Backbone: pretrained ResNet-50
        self.backbone = resnet50(pretrained=True)
        # Remove the last fully connected layer and avgpool
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # ResNet outputs 2048 channels, we'll project this to hidden_dim
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        
        # Transformer
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        
        # Object queries (learned positional embeddings for decoder)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))
        
        # Spatial positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        
        # Prediction heads
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object" class
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
    
    def forward(self, x):
        # 1. Extract features from backbone
        features = self.backbone(x)  # [batch_size, 2048, h, w]
        
        # 2. Project features to hidden_dim
        h = self.conv(features)  # [batch_size, hidden_dim, h, w]
        batch_size, _, height, width = h.shape
        
        # 3. Flatten spatial dimensions and permute for transformer
        h = h.flatten(2).permute(2, 0, 1)  # [h*w, batch_size, hidden_dim]
        
        # 4. Create positional encodings
        pos = torch.cat([
            self.col_embed[:width].unsqueeze(0).repeat(height, 1, 1),
            self.row_embed[:height].unsqueeze(1).repeat(1, width, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # [h*w, 1, hidden_dim]
        
        # 5. Transformer forward pass
        query_pos = self.query_pos.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries, batch_size, hidden_dim]
        tgt = torch.zeros_like(query_pos)
        
        h = self.transformer(
            src=h,
            tgt=tgt,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            pos=pos,
            query_pos=query_pos
        )  # [num_queries, batch_size, hidden_dim]
        
        # 6. Final prediction heads
        outputs_class = self.linear_class(h)  # [num_queries, batch_size, num_classes + 1]
        outputs_coord = self.linear_bbox(h).sigmoid()  # [num_queries, batch_size, 4]
        
        # Transpose to [batch_size, num_queries, ...] for convenience
        out = {'pred_logits': outputs_class.transpose(0, 1),
               'pred_boxes': outputs_coord.transpose(0, 1)}
        
        return out


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    
    For efficiency reasons, the targets don't include the no_object. Because of this, 
    in general, there are more predictions than targets. In this case, we do a 1-to-1 
    matching of the best predictions, while the others are un-matched (and thus treated as non-objects).
    """
    
    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten batch dimension
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Compute the classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the giou cost
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes))
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Compute all the requested losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses


# Helper functions for box operations
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    
    # Find the smallest enclosing box
    lt_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_max = (rb_max - lt_min).clamp(min=0)  # [N,M,2]
    area_max = wh_max[:, :, 0] * wh_max[:, :, 1]
    
    return iou - (area_max - union) / area_max


# Example usage
if __name__ == "__main__":
    # Create model
    num_classes = 91  # COCO number of classes
    num_queries = 100
    model = DETR(num_classes=num_classes, num_queries=num_queries)
    
    # Example input
    x = torch.rand(2, 3, 800, 800)  # batch of 2 RGB images of size 800x800
    outputs = model(x)
    
    print("Output shapes:")
    print(f"Class predictions: {outputs['pred_logits'].shape}")  # Should be [2, 100, 92]
    print(f"Box predictions: {outputs['pred_boxes'].shape}")     # Should be [2, 100, 4]
    
    # Example targets (for loss computation)
    targets = [
        {
            "labels": torch.tensor([10, 20]),  # class labels
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.7, 0.1, 0.1]])  # cxcywh format, normalized
        },
        {
            "labels": torch.tensor([15]),
            "boxes": torch.tensor([[0.4, 0.6, 0.3, 0.3]])
        }
    ]
    
    # Create matcher and criterion
    #matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
    #weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
    #criterion = SetCriterion(num_classes, matcher, weight_dict)
    
    # Compute loss
    #losses = criterion(outputs, targets)
    #print("\nLosses:")
    #for k, v in losses.items():
    #    print(f"{k}: {v.item()}")
