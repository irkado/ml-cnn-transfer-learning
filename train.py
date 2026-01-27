import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import TransferModel  # your TransferModel class

# -------------------------
# 1. Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
# -------------------------
# 2. Transforms
# CIFAR-10 is 32x32, ResNet18 needs 224x224
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),          # flip image horizontally with 50% chance
    transforms.RandomRotation(10),              # rotate image ±10 degrees
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2
    ),                                          # small color variations
    transforms.ToTensor(),
])


# -------------------------
# 3. Dataset & DataLoader (small subset for quick test)
# -------------------------
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# Use only a subset to test quickly
# train_dataset = Subset(train_dataset, range(2000))  # first 2000 images
# test_dataset = Subset(test_dataset, range(500))     # first 500 images

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Datasets loaded")

# -------------------------
# 4. Model
# -------------------------
model = TransferModel(
    num_classes=10,
    backbone="resnet18",
    pretrained=True,
    dropout=0.4
)
# -------------------------
# FREEZE/UNFREEZE LAYERS
# -------------------------
# Options:
# 1) Head only
# model.train_head_only()

# 2) Fine-tune last block + head
#model.fine_tune_last_block()

# 3) Full fine-tuning
model.unfreeze_all()

model = model.to(device)

# -------------------------
# 5. Loss & Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.get_trainable_params(), lr=1e-5)

# -------------------------
# 6. Training Loop
# -------------------------
epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:  # print every 10 batches
            print(f"Batch {i}/{len(train_loader)} processed")

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

# -------------------------
# 8. Save model
# -------------------------
torch.save(model.state_dict(), "transfer_model.pth")
print("Model saved as transfer_model.pth")
