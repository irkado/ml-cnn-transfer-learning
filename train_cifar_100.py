import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random, numpy as np, pandas as pd

from model import TransferModel
from data_preparation.CIFAR_100.data_cifar_100 import train_loader, validation_loader, test_loader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# might try using these for reproducibility later not sure tho
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def evaluate(model: TransferModel, loader: DataLoader, final_report=False) -> tuple:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    loss_list, true_labels, predicted_labels = [], [], []

    criterion = nn.CrossEntropyLoss()

    # disable gradient tracking for memory and speed eval purposes
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            loss_list.append(loss.item())

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if final_report:
                true_labels.extend(labels.cpu().numpy())  # for confusion matrix also better to save
                predicted_labels.extend(preds.cpu().numpy())

    if final_report:  # report will be used only during final test evaluation
        return loss_sum / len(loader), correct / total, loss_list, true_labels, predicted_labels
    else:
        return loss_sum / len(loader), correct / total  # only return loss and accuracy


def train_phase(model: TransferModel, epochs: int, lr: float, phase_name: str, save_name: str) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.get_trainable_params(), lr=lr)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}  # save metrics for plots

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)  # less memory
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, validation_loader)

        # save for each epoch train_loss, val_loss, val_acc
        history['train_loss'].append(train_loss)  # for overfitting underfitting graphs
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"{phase_name}: Epoch {epoch + 1}/{epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(f"saved best -> {save_name} (val_acc={best_val_acc * 100:.2f}%)")

    # save history to csv
    df_history = pd.DataFrame.from_dict(history, orient='index').transpose()
    df_history.to_csv(f"{save_name.replace('.pth', '')}_training_history.csv", index=False)
    return best_val_acc


def run_training(backbone: str) -> None:
    # train clasifier head only
    # freeze backbone, higher lr to adapt to 100 classes
    model = TransferModel(num_classes=100, backbone=backbone, pretrained=True, dropout=0.4).to(device)
    model.train_head_only()
    train_phase(model, epochs=5, lr=1e-3, phase_name=f"{backbone}: head",
                save_name=f"{backbone}_cifar100_head_best.pth")

    # fine-tune last backbone block and classifier
    # adapt higher level features, lower lr (for pretrained weights)
    model = TransferModel(num_classes=100, backbone=backbone, pretrained=True, dropout=0.3).to(device)
    model.load_state_dict(torch.load(f"{backbone}_cifar100_head_best.pth", map_location=device))
    model.fine_tune_last_block()
    train_phase(model, epochs=5, lr=1e-4, phase_name=f"{backbone}: lastblock",
                save_name=f"{backbone}_cifar100_lastblock_best.pth")

    # fine tune full
    # unfreeze all weights, very low lr (again for the pretrained weights)
    model = TransferModel(num_classes=100, backbone=backbone, pretrained=True, dropout=0.2).to(device)
    model.load_state_dict(torch.load(f"{backbone}_cifar100_lastblock_best.pth", map_location=device))
    model.unfreeze_all()
    train_phase(model, epochs=15, lr=1e-5, phase_name=f"{backbone}: full",
                save_name=f"{backbone}_cifar100_full_best.pth")

    # final eval on test set (imprtant for Ira)
    model.load_state_dict(torch.load(f"{backbone}_cifar100_full_best.pth", map_location=device))
    test_loss, test_acc, loss_list, true_labels, predicted_labels = evaluate(model, test_loader, final_report=True)

    # save loss list separately from final test eval
    df_batch_losses = pd.DataFrame(loss_list, columns=['batch_loss'])
    df_batch_losses.to_csv(f"{backbone}_test_loss_list.csv", index=False)

    # save true vs predicted labels for confusion matrix later from final test eval
    df_predictions = pd.DataFrame({
        'True_Label': true_labels,
        'Predicted_Label': predicted_labels
    })
    df_predictions.to_csv(f"{backbone}_test_predictions.csv", index=False)

    print(f"{backbone} testing: loss={test_loss:.4f} acc={test_acc * 100:.2f}%")


# might take a while, feel free to comment out either one if u want just one
if __name__ == "__main__":
    run_training("resnet18")
    run_training("mobilenet_v2")
