# Machine Learning Project: Develop a CNN for Image Classification using Transfer Learning

**Team Members**: Ekaterina Nedeva, Hanane Lotf, Iryna Dolenko, Yaroslav Narozhnyi

This repository contains a CNN model realisation for classifying across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Model was trained on the data from CIFAR-10 dataset.

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

## Workload Distribution
**Yarik** - Data Preparation: Load and preprocess the CIFAR-10 dataset, perform normalization, resizing, and data augmentation, and split data into training, validation, and test sets


**Kate**  -  Model Design: Select the pre-trained CNN architecture, modify the classifier head for CIFAR-10, configure loss function and optimization strategy


**Hanane** - Training: Train the model using transfer learning, experiment with freezing/unfreezing layers, tune hyperparameters


**Ira** - Evaluation: Evaluate model performance using accuracy and loss, analyze confusion matrix and class-wise performance