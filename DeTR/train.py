import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import requests
from io import BytesIO

from model import DeTR 

# Initialize model (using the full implementation from earlier)
model = DeTR(num_classes=91)  # COCO classes
model.eval()

# Load sample image
url = "https://images.unsplash.com/photo-1608848461950-0fe51dfc41cb?q=80&w=1000"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

# Preprocess image (same as DETR validation transforms)
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Run model
with torch.no_grad():
    outputs = model(img_tensor)

# Process outputs
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Remove "no object" class
keep = probas.max(-1).values > 0.7  # Confidence threshold

# Convert boxes from [0;1] to image scales
img_w, img_h = img.size
bboxes_scaled = outputs['pred_boxes'][0, keep].cpu().numpy()
bboxes_scaled[:, [0, 2]] *= img_w
bboxes_scaled[:, [1, 3]] *= img_h

# COCO classes (simplified)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella'
]

# Visualization function
def plot_results(pil_img, prob, boxes, classes=COCO_CLASSES):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    for p, (x, y, w, h) in zip(prob, boxes):
        cl = p.argmax()
        class_name = classes[cl]
        confidence = p[cl]
        
        # Create rectangle
        rect = patches.Rectangle(
            (x-w/2, y-h/2), w, h, 
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        plt.text(
            x-w/2, y-h/2, f'{class_name}: {confidence:.2f}',
            color='white', fontsize=12,
            bbox=dict(facecolor='red', alpha=0.5) 
        )

    plt.axis('off')
    plt.show()

# Get top predictions
top_probas = probas[keep].cpu().numpy()
top_boxes = bboxes_scaled

# Show results
plot_results(img, top_probas, top_boxes)
