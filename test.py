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

test_data = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    download=True,
    transform=transform,
)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle=False, num_workers=0)
print("Test dataset size:", len(test_data))

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) #128 - 3 = 125 / 1 + 1 = 126, [16, 126, 126]
        self.pool = nn.MaxPool2d(2, 2) #[16, 63, 63]

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) #63 - 3 = 60 / 1 + 1 = 61, [32, 61, 61] -> [32, 30, 30]
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) #30 - 3 = 27 / 1 + 1 = 28, [64, 28, 28] -> [64, 14, 14]

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) #14 - 3 = 11 / 1 + 1 = 12, [128, 12, 12] -> [128, 6, 6]

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 37)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        
        x = self.fc3(x)
        return x



net = NeuralNet()
net.load_state_dict(torch.load("trained_model.pth"))
correct = 0
total = 0

net.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")
