# Fine Tune Vit

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Load pre-trained ViT model
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=1000)  # Adjust num_labels as needed

# Prepare ImageNet dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="path/to/imagenet/train", transform=transform)
val_dataset = datasets.ImageFolder(root="path/to/imagenet/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    scheduler.step()
    print(f"Epoch {epoch+1} completed")

# Save the fine-tuned model
model.save_pretrained("vit-imagenet-finetuned")

