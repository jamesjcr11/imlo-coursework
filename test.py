import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.datasets as datasets
import copy
from torchvision.transforms import v2
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

torch.manual_seed(7)
np.random.seed(7)   



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


eval_transform = transforms.Compose ([
    transforms.Resize((200, 200)),
    transforms.CenterCrop(180),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])



test_data = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    download=True,
    transform=eval_transform,
)


test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle=False, num_workers=0)



class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, do_pool=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.skip = nn.Identity()
        if in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_c)
            )

        self.do_pool = do_pool
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.skip(x)

        out = F.silu(self.bn1(self.conv1(x)))
        out = F.silu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + identity
        out = F.silu(out)

        if self.do_pool:
            out = self.pool(out)

        return out


#///////////////////////////////////////////////////////////////


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 384)
        self.conv5 = ConvBlock(384, 512)

        self.gap = nn.AdaptiveAvgPool2d((2,2))

        self.fc1 = nn.Linear(512 * 2 * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 37)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = F.silu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.fc2(x)
        return x
    
    

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



net = NeuralNet().to(device)
net.load_state_dict(torch.load("model.pth", map_location=device))

test_acc = get_accuracy(net, test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
