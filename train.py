import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import TransferModel 
import random
import numpy as np
import torch

SEED = 42  


random.seed(SEED)


np.random.seed(SEED)


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)


#train_dataset = Subset(train_dataset, range(2000))  
#test_dataset = Subset(test_dataset, range(500))     

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Datasets loaded")

model = TransferModel(
    num_classes=10,
    backbone="resnet18",
    pretrained=True,
    dropout=0.2
)

model.train_head_only()

#model.fine_tune_last_block()

#model.unfreeze_all()

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.get_trainable_params(), lr=1e-3)

epochs = 5

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

        if i % 10 == 0:  
            print(f"Batch {i}/{len(train_loader)} processed")

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")


torch.save(model.state_dict(), "transfer_model_10.pth")
print("Model saved as transfer_model_10.pth")
