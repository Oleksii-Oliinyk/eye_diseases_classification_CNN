import torch
import torch.nn.functional as F
import torch.nn as nn

class EyeDiseaseClassifierCNN(nn.Module):
    def __init__(self):
        super(EyeDiseaseClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        
        self.dropout1 = nn.Dropout(0.35)
        #self.dropout2 = nn.Dropout(0.15)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(80 * 80 * 32, 1024)
        self.fc2 = nn.Linear(1024, 8)
        #self.fc3 = nn.Linear(256, 8)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)       
        return x
        
        
        
        