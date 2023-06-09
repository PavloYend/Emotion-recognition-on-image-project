
import torch.nn as nn

class EmotionClassifier_v1(nn.Module):
    def __init__(self):
        super(EmotionClassifier_v1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(18432, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 7)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = x.flatten()

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class EmotionClassifier_v3(nn.Module):
    def __init__(self):
        super(EmotionClassifier_v3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(73728, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = x.flatten()

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x