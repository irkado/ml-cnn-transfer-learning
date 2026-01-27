import torch
import torchvision
import numpy as np
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader


SEED = 42

random.seed(SEED)  
np.random.seed(SEED) 
torch.manual_seed(SEED) # CPU
torch.cuda.manual_seed(SEED) # GPU
torch.cuda.manual_seed_all(SEED) # GPU


train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_full = datasets.CIFAR10("./Data_Preparation/CIFAR_10/cifar_10_data", train=True, download=True, transform=train_tfms)
test_set = datasets.CIFAR10("./Data_Preparation/CIFAR_10/cifar_10_data", train=False, download=True, transform=test_tfms)

val_size = 5000
train_size = len(train_full) - val_size
train_set, validation_set = torch.utils.data.random_split(
    train_full,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

batch_size = 64

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,generator=torch.Generator().manual_seed(SEED))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=2)