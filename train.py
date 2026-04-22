import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms 
import torchvision.datasets as datasets


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_data = train_data = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    download=True,
    transform=transform,
)

test_data = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    download=True,
    transform=transform,
)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 32, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle=False, num_workers=2)

print("Train dataset size:", len(train_data))
print("Test dataset size:", len(test_data))

image, label = train_data[0]
print(image.size())

class_names = train_data.dataset.classes
print("Class names:", class_names)