import torch.nn.functional as F
import torch.nn as nn
from torch import flatten

class RetinoblastomaClassifierNonBinaryCNN(nn.Module):
    def __init__(self):
        super(RetinoblastomaClassifierNonBinaryCNN, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)
        
        x = flatten(x, start_dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x