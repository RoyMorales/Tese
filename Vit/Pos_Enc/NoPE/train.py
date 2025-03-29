# Train Model

import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

from model import VisionTransformer

device = (
        "cuda" if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )

print (f"\n Using {device} device \n")

###############################################

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    trainning_data = datasets.CIFAR10(
            root='data',
            train=True,
            download=True,
            transform=transform
            )

    test_data = datasets.CIFAR10(
            root='data',
            train=False,
            download=True,
            transform=transform
            )

    train_dataloader = DataLoader(trainning_data, batch_size=batch_size, shuffle=True)
    test_dataloader =DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader

def train(dataloader, model, loss_fn, optmizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)


        optmizer.zero_grad()
        loss.backward()
        optmizer.step()


        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg Loss: {test_loss:>8f}\n")

if __name__ == '__main__':
    batch_size = 32
    epochs = 10
    num_class = 10

    # Nvidia GuideLine
    torch.backends.cudnn.benchmark = True

    train_dataloader, test_dataloader = load_data()
    
    model = VisionTransformer(img_size=32, patch_size=4, in_channels=3, num_classes=num_class, embed_dim=256, depth=4, num_heads=4)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optmizer = torch.optim.AdamW(model.parameters(), lr= 1e-3)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n ------------------------------")
        train(train_dataloader, model, loss_fn, optmizer)
        test(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), "Vit.pth")
    print("Saved Pytorch Model State to Vit.pth")



