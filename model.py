import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        '''
        Input shape: (bs,c,h,w) #(bs,4,84,84)
        Output shape: (bs,n_actions)
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
# Dueling DQN
class DuelDQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.Alinear1 = nn.Linear(64*7*7,128)
        self.Alinear2 = nn.Linear(128,n_actions)
        self.Vlinear1 = nn.Linear(64*7*7,128)
        self.Vlinear2 = nn.Linear(128,1)
        
    def forward(self, x):
        '''
        Input shape: (bs,c,h,w) #(bs,4,84,84)
        Output shape: (bs,n_actions)
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x,1)

        Ax = F.leaky_relu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)

        Vx = F.leaky_relu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)
        
        return Vx + (Ax - Ax.mean())