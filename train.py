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


train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 32, shuffle=False, num_workers=0)

print("Train dataset size:", len(train_data))

image, label = train_data[0]
print(image.size())

class_names = train_data.dataset.classes
print("Class names:", class_names)


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


def get_accuracy(model, loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return 100 * correct / total


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
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(30):
    print(f"Training epoch {epoch} ...")

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_acc = get_accuracy(net, train_loader)
    val_acc = get_accuracy(net, val_loader)
    print(f"Loss: {running_loss / len(train_loader):.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")


torch.save(net.state_dict(), 'trained_model.pth')