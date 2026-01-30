import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch
from torch import optim 
from tqdm import tqdm

from model import TransferModel 
from Data_Preparation.CIFAR_10.data_cifar_10 import train_loader, validation_loader, test_loader

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

# Yarik
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    print("Evaluation started")
    loop = tqdm(loader, desc="Evaluation")
    with torch.no_grad():
        for i,(images, labels) in enumerate(loop):
            loop.set_description(f"Processing instance {i}")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(loader), correct / total

def train_phase(model, epochs, lr, phase_name, save_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.get_trainable_params(), lr=lr)

    train_loss_list, validation_loss_list, val_accuracy_list = [], [], []

    best_val_acc = 0.0
    print(f"\n\n\n {phase_name} started")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=phase_name, leave=False)

        for i, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True) # less memory
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_loss_list.append(train_loss)
        
        
        val_loss, val_acc = evaluate(model, validation_loader)
        validation_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)

        print(f"{phase_name}: Epoch {epoch+1}/{epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(f"saved best -> {save_name} (val_acc={best_val_acc*100:.2f}%)")

    return best_val_acc,  train_loss_list, validation_loss_list , val_accuracy_list

def run_training(backbone: str):
    # train classifier head only
    # freeze backbone, higher lr to adapt to 10 classes
    model = TransferModel(num_classes=10, backbone=backbone, pretrained=True, dropout=0.4).to(device)
    model.train_head_only()
    train_phase(model, epochs=5, lr=1e-3, phase_name=f"{backbone}: head", save_name=f"{backbone}_cifar10_head_best.pth")

    # fine-tune last backbone block and classifier
    # adapt higher level features, lower lr (for pretrained weights)
    model = TransferModel(num_classes=10, backbone=backbone, pretrained=True, dropout=0.3).to(device)
    model.load_state_dict(torch.load(f"{backbone}_cifar10_head_best.pth", map_location=device))
    model.fine_tune_last_block()
    train_phase(model, epochs=5, lr=1e-4, phase_name=f"{backbone}: lastblock", save_name=f"{backbone}_cifar10_lastblock_best.pth")

    # fine tune full
    # unfreeze all weights, very low lr (again for the pretrained weights)
    model = TransferModel(num_classes=10, backbone=backbone, pretrained=True, dropout=0.2).to(device)
    model.load_state_dict(torch.load(f"{backbone}_cifar10_lastblock_best.pth", map_location=device))
    model.unfreeze_all()
    train_phase(model, epochs=15, lr=1e-5, phase_name=f"{backbone}: full", save_name=f"{backbone}_cifar10_full_best.pth")

    # final eval on test set (important for Ira)
    model.load_state_dict(torch.load(f"{backbone}_cifar10_full_best.pth", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"{backbone} testing: loss={test_loss:.4f} acc={test_acc*100:.2f}%")

if __name__ == "__main__":
    run_training("resnet18")
    run_training("mobilenet_v2")