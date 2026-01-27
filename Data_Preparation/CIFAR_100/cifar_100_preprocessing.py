import torch
import torchvision
import random
import numpy as np
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


train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224,padding=8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR100("./Data_Preparation/CIFAR_10/cifar_100_data", train=True,transform=train_transforms,download=True)
test_dataset = torchvision.datasets.CIFAR100("./Data_Preparation/CIFAR_10/cifar_100_data", train=False,transform=test_transforms,download=True)

validation_size = int(len(train_dataset) * 0.1)

train_set, validation_set =  torch.utils.data.random_split(train_dataset,lengths=[len(train_dataset) - validation_size, validation_size], generator=torch.Generator().manual_seed(SEED))

print(f"The size of training set is {len(train_set)} samples")
print(f"The size of validation set is {len(validation_set)} samples")

batch_size = 64


validation_loader = DataLoader(validation_set,batch_size=batch_size,shuffle=False,num_workers=2)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=2, generator=torch.Generator().manual_seed(SEED))
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)