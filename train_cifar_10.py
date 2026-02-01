import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

from model import TransferModel
from data_processing import train_loader, validation_loader, test_loader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# for reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Datasets loaded")

def evaluate(model: TransferModel, loader, final_report: bool = False): #eval func we will use to run the model on validation (every epoch) and for testing at the end.
    model.eval() #dropout disabled all neurons active + BatchNorm uses running stats for testing
    #evaluation variables
    correct = 0 #correct predictions
    total = 0  #total samples evaluated (updated each batch)
    loss_sum = 0.0 #total loss across all batches
    loss_list, true_labels, predicted_labels = [], [], [] #loss_list: storing loss per batch (for plotting), true labels and predicted labels for confusion matrice
    # loss function for evaluation
    criterion = nn.CrossEntropyLoss() # 1.softmax(converts logits the model outputs into probabilities) 2.puniches model when correct class has low probability (loss=-log(predicted prob))

    # disable gradient tracking for memory and speed
    with torch.no_grad(): #no weight change--> faster with less memory usage 
        loop = tqdm(loader, desc="Evaluation", leave=False) if not final_report else loader #dont wrap loader in tqdm for  final report 

        for images, labels in loop: #looping over batches
            images, labels = images.to(device), labels.to(device) # send data to gpu (needs to be same device as models)
            outputs = model(images) #logits from the model
            loss = criterion(outputs, labels) #getting loss using criterion
            loss_sum += loss.item() #getting sum of losses of all batches
            loss_list.append(loss.item()) #keeping track of loss per batch (for plots)

            preds = outputs.argmax(dim=1) #pick class with highest probability for each sample
            correct += (preds == labels).sum().item() #number of correct prediction
            total += labels.size(0) #samples processed so far

            if final_report: #keeping tabs of true labels and predicted labels for confusion matrix (false by default)
                true_labels.extend(labels.cpu().numpy().tolist())
                predicted_labels.extend(preds.cpu().numpy().tolist())

    avg_loss = loss_sum / len(loader) #sum losses devided by sum number of batches
    acc = correct / total


    if final_report:
        return avg_loss, acc, loss_list, true_labels, predicted_labels #return all data for testing
    return avg_loss, acc #only these for validation


# added csv saving
def train_phase(model: TransferModel, epochs: int, lr: float, phase_name: str, save_name: str) -> float: #we take all hyper parameters as arguments (+phase and save name)
                                                                                                               #and we return best val acc
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda")) #scale gradients up so that they can be represented by fp16 while avoiding underflow
    criterion = nn.CrossEntropyLoss() #loss function for training
    optimizer = optim.Adam(model.get_trainable_params(), lr=lr) #adaptive optimizer updates weights based on the gradient of the loss with respect to each trainable parameter


    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}  # save metrics for plots

    print(f"\n\n\n {phase_name} started")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0 #running loss intialised for each epoch
        loop = tqdm(train_loader, desc=phase_name, leave=False) # for tracking progress of each epoch

        for i, (images, labels) in enumerate(loop): #iterate over each batch
            images, labels = images.to(device), labels.to(device) # move training set this time to gpu

            optimizer.zero_grad(set_to_none=True) # clears old gradients (none instead of zero)-->less memory and little faster
            outputs = model(images)
            loss = criterion(outputs, labels) # compute loss using loss function (cross entropy between model output and true labels)
            loss.backward() # loss gradient computation
            optimizer.step() # update weight using gradient
            loop.set_postfix(loss=loss.item()) # updates progress bar display with current batch loss

            running_loss += loss.item() # sum of all losses in this batch

# in case training is really slow, comment out the above loop and uncomment below (only works for cuda)
        # for i, (images, labels) in enumerate(loop):
        #     images, labels = images.to(device), labels.to(device)
        #     optimizer.zero_grad(set_to_none=True)  # less memory
        #     with torch.cuda.amp.autocast(enabled=(device.type == "cuda")): #autocast for automatically choosing precision per operator 
        #         outputs = model(images)
        #         loss = criterion(outputs, labels)
        #
        #     scaler.scale(loss).backward() # we scale loss
        #     scaler.step(optimizer) # unscale gradient
        #     scaler.update() #increase/decrease scale depending on if there is overflow
        #
        #     loop.set_postfix(loss=loss.item())
        #     running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, validation_loader)

        # save for each epoch
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"{phase_name}: Epoch {epoch+1}/{epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(f"saved best -> {save_name} (val_acc={best_val_acc*100:.2f}%)")

    # save history to csv
    df_history = pd.DataFrame.from_dict(history, orient='index').transpose()
    df_history.to_csv(f"{save_name.replace('.pth', '')}_training_history.csv", index=False)

    return best_val_acc


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

    # final eval on test set
    model.load_state_dict(torch.load(f"{backbone}_cifar10_full_best.pth", map_location=device))
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

    print(f"{backbone} testing: loss={test_loss:.4f} acc={test_acc*100:.2f}%")


if __name__ == "__main__":
    #run_training("resnet18")
    run_training("mobilenet_v2")
