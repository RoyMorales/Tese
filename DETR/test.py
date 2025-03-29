import torch
import torch.nn as nn
import torch.nn.functional as F

# Bottleneck Block for ResNet-50
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNet-50 Backbone
class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64

        # Initial layers
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

# DETR Model
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # Backbone (ResNet-50)
        self.backbone = ResNet50Backbone()
        self.conv = nn.Conv2d(2048, hidden_dim, 1)  # Project backbone features to hidden_dim

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

        # Prediction heads
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # Query embeddings
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # Positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        h = self.conv(features)
        bs, c, h_dim, w_dim = h.shape

        # Flatten spatial dimensions
        h = h.flatten(2).permute(2, 0, 1)  # (H*W, bs, hidden_dim)

        # Positional encodings
        row_embed = self.row_embed[:h_dim].unsqueeze(1).repeat(1, w_dim, 1)  # (H, W, hidden_dim // 2)
        col_embed = self.col_embed[:w_dim].unsqueeze(0).repeat(h_dim, 1, 1)  # (H, W, hidden_dim // 2)
        pos = torch.cat([row_embed, col_embed], dim=-1).flatten(0, 1).unsqueeze(1)  # (H*W, 1, hidden_dim)

        # Transformer
        h = self.transformer(pos + h, self.query_pos.unsqueeze(1).repeat(1, bs, 1))

        # Prediction heads
        outputs_class = self.linear_class(h)
        outputs_coord = self.linear_bbox(h).sigmoid()

        return {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

# Example usage
if __name__ == "__main__":
    model_resnet = ResNet50Backbone()
    y = torch.rand(2, 3, 224, 224)
    outputs = model_resnet(y)
    print(outputs)


    model = DETR(num_classes=91)  # COCO has 80 classes + background
    x = torch.rand(2, 3, 224, 224)  # Batch of 2 images
    outputs = model(x)
    print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
