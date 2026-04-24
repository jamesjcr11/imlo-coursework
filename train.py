import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
import copy

torch.manual_seed(42)
np.random.seed(42)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    #transforms.RandomRotation(5),
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])  


train_data =  datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    download=True,
    transform=train_transform,
)


test_data = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    download=True,
    transform=test_transform,
)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle=False, num_workers=0)




train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64, shuffle=False, num_workers=0)



image, label = train_data[0]
class_names = train_data.dataset.classes




def get_accuracy(model, loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)     
            labels = labels.to(device)     

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return 100 * correct / total




class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),     
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
  
        self.pool = nn.MaxPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(256 * 4 * 4,  1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 37)

        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
    
                
        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x)

        x = self.fc3(x)
        return x
    

net = NeuralNet().to(device)
#loss_function = nn.CrossEntropyLoss(label_smoothing=0.05) 
loss_function = nn.CrossEntropyLoss()      
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_val_acc = 0.0
best_state_dict = None

for epoch in range(30):
    print(f"Training epoch {epoch} ...")

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print(i)
      
    train_acc = get_accuracy(net, train_loader)
    val_acc = get_accuracy(net, val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(best_val_acc)
        best_state_dict = copy.deepcopy(net.state_dict())
        print(f"Saved best model with val acc: {val_acc:.2f}%")
    print(f"Loss: {running_loss / len(train_loader):.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("\n")

    scheduler.step()
torch.save(best_state_dict, 'trained_model.pth')
net.load_state_dict(torch.load("trained_model.pth", map_location=device))
val_acc_loaded = get_accuracy(net, val_loader)

print(f"Best val accuracy recorded: {best_val_acc:.2f}%")
print(f"Validation accuracy after loading: {val_acc_loaded:.2f}%")
test_acc = get_accuracy(net, test_loader)

print(f"Test Accuracy: {test_acc:.2f}%")