import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


'''
Net: 
        Constructs a classifier which takes images of input size (32,32,3)
Args:
        classes: no of classes
'''

class Net(nn.Module):
    def __init__(self, classes=6):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 *5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
#         print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
#         print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
#         print(x.shape)
        x = x.view(-1, 16 *5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
