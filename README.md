# Machine Learning Project: Develop a CNN for Image Classification using Transfer Learning

**Team Members**: Ekaterina Nedeva, Hanane Lotf, Iryna Dolenko, Yaroslav Narozhnyi

This repository contains a CNN model realisation for classifying across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Model was trained on the data from CIFAR-10 (and CIFAR-100) dataset(s).

The project was developed using **Python 3.12**.

---

## Project Setup

1. Ensure that python version 3.12 is installed.
2. Clone the repository:
```bash
git clone https://github.com/irkado/ml-cnn-transfer-learning.git
cd ml-cnn-transfer-learning
```
3. To avoid conflicts with dependencies project uses venv virtual environment.<br>
**To create the environment:**
```bash
python3.12 -m venv .venv
```
**Activate the environment:**

```bash
source .venv/bin/activate
```
**Install dependencies (.venv must be activated):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**To exit environment**
```bash
deactivate
```
---
add new library to .venv envi
-> ```pip install scikit-learn```<br>
update requirements.txt
-> ```pip freeze > requirements.txt```

## Dataset download / first run

On the first run, the dataset will be downloaded automatically by the PyTorch dataset utilities (no manual dataset download step required). 
Make sure to run the script again on a network that allows dataset downloads or download once on a different machine and copy the dataset cache.

## Training

If a CUDA-capable NVIDIA GPU + drivers are available, PyTorch will use the GPU; otherwise it will fall back to CPU (not recommended :))

The training scripts handle model initialization, dataset loading, training, validation, saving the best model automatically (model weights saved as `.pt(h)` files).

To train run `python train_cifar_10.py` or `python train_cifar_100.py` respectively. 

### Saved Files
For each backbone (`resnet18` and `mobilenet_v2`) the script produces these files (analogically for CIFAR-10):
1. `BACKBONE_cifar100_head_best.pth`: contains PyTorch `state_dict` (weights only) of the model that achieved the highest validation accuracy during the *head-only* phase.  

2. `BACKBONE_cifar100_lastblock_best.pth`: contains PyTorch `state_dict` (weights only) of the model that achieved the highest validation accuracy during the *fine-tune last block + classifier* phase.  

3. `BACKBONE_cifar100_full_best.pth`: contains PyTorch `state_dict` (weights only) of the model that achieved the highest validation accuracy during the *unfreeze all / full fine-tune* phase.

4. `BACKBONE_cifar100_head_best_training_history.csv`  (and similarly for lastblock/full): contains per-epoch metrics for that specific phase

	Columns:
	- `train_loss`: average training loss per epoch (`running_loss / len(train_loader)`)
	- `val_loss`: average validation loss per epoch (`loss_sum / len(val_loader)`)
	- `val_acc`: validation accuracy per epoch (`correct/total`)
	
	Rows: one row per epoch in that phase.
	- `BACKBONE_cifar100_head_best_training_history.csv` (5 rows)
	- `BACKBONE_cifar100_lastblock_best_training_history.csv` (5 rows)
	- `BACKBONE_cifar100_full_best_training_history.csv` (15 rows)
	
5. `BACKBONE_test_loss_list.csv`: contains: batch-level test losses from the final test evaluation only (after loading `BACKBONE_cifar100_full_best.pth`).  
	- column: `batch_loss`
	- one row per test batch (length = number of batches in `test_loader`)
	This is used to plot how loss varies across test batches (not per-epoch and not per-sample).
	
6. `BACKBONE_test_predictions.csv`: contains sample-level predictions from the final test evaluation only (again using the final full fine-tuned checkpoint).  
	- `True_Label`: integer class index from the dataset
	- `Predicted_Label`: integer class index predicted by the model
	This is what used for a confusion matrix and classification report.

## Workload Distribution
**Yarik** - Data Preparation: Load and preprocess the CIFAR-10 dataset, perform normalization, resizing, and data augmentation, and split data into training, validation, and test sets


**Kate**  -  Model Design: Select the pre-trained CNN architecture, modify the classifier head for CIFAR-10, configure loss function and optimization strategy


**Hanane** - Training: Train the model using transfer learning, experiment with freezing/unfreezing layers, tune hyperparameters


**Ira** - Evaluation: Evaluate model performance using accuracy and loss, analyze confusion matrix and class-wise performance