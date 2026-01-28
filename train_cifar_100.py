import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from model import TransferModel
from data_preparation.CIFAR_100.data_cifar_100 import train_loader, validation_loader, test_loader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#might try using these for reproducibility later not sure tho
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

'''Ira lemme know what u think abt this eval'''
#added this so we can save the best model, not just the last one
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    #disable gradient tracking for memory and speed eval purposes
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(loader), correct / total
'''eval ends here'''

def train_phase(model, epochs, lr, phase_name, save_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.get_trainable_params(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True) #less memory
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, validation_loader)

        print(f"{phase_name}: Epoch {epoch+1}/{epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(f"saved best -> {save_name} (val_acc={best_val_acc*100:.2f}%)")

    return best_val_acc


def run_training(backbone: str):

    # train clasifier head only
    # freeze backbone, higher lr to adapt to 100 classes
    model = TransferModel(num_classes=100, backbone=backbone, pretrained=True, dropout=0.4).to(device)
    model.train_head_only()
    train_phase(model, epochs=5, lr=1e-3, phase_name=f"{backbone}: head", save_name=f"{backbone}_cifar100_head_best.pth")

    # fine-tune last backbone block and classifier
    # adapt higher level features, lower lr (for pretrained weights)
    model = TransferModel(num_classes=100, backbone=backbone, pretrained=True, dropout=0.3).to(device)
    model.load_state_dict(torch.load(f"{backbone}_cifar100_head_best.pth", map_location=device))
    model.fine_tune_last_block()
    train_phase(model, epochs=5, lr=1e-4, phase_name=f"{backbone}: lastblock", save_name=f"{backbone}_cifar100_lastblock_best.pth")

    # fine tune full
    # unfreeze all weights, very low lr (again for the pretrained weights)
    model = TransferModel(num_classes=100, backbone=backbone, pretrained=True, dropout=0.2).to(device)
    model.load_state_dict(torch.load(f"{backbone}_cifar100_lastblock_best.pth", map_location=device))
    model.unfreeze_all()
    train_phase(model, epochs=15, lr=1e-5, phase_name=f"{backbone}: full", save_name=f"{backbone}_cifar100_full_best.pth")

    '''check this one too pls'''
    # final eval on test set (imprtant for Ira)
    model.load_state_dict(torch.load(f"{backbone}_cifar100_full_best.pth", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"{backbone} testing: loss={test_loss:.4f} acc={test_acc*100:.2f}%")

#might take a while, feel free to comment out either one if u want just one
if __name__ == "__main__":
    run_training("resnet18")
    run_training("mobilenet_v2")
