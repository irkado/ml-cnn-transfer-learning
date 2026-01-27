import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch

from model import TransferModel 
from data_processing import train_loader, validation_loader, test_loader

SEED = 42  
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
