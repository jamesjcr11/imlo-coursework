import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

torch.manual_seed(7)
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((200,200)),
    transforms.CenterCrop(180),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])


train_data =  datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    download=True,
    transform=train_transform,
)


train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True, num_workers=0)



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
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)



best_val_acc = 0.0
best_state_dict = None

for epoch in range(30):
    print(f"Training epoch {epoch  + 1} ...")

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    print(f"Training Loss: {running_loss / len(train_loader):.4f}")
    train_acc = get_accuracy(net, train_loader)
    print(f"Train Accuracy: {train_acc:.2f}%")
    print("\n")

    scheduler.step()

torch.save(net.state_dict(), "model.pth")
print(f"Final Train Accuracy: {train_acc:.2f}%")