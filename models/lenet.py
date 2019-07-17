import torch.nn as nn
import torch.nn.functional as F 

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):               
        #print(x.size())
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = F.max_pool2d(x, 2, 2)
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = F.max_pool2d(x, 2, 2)
        #print(x.size())
        x = x.view(-1, 5*5*50)
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x
    
    def name(self):
        return "LeNet"

class CustomNet(nn.Module):

    def __init__(self):
        """
            Inicializar capas de la red
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
            Procesar entrada x a traves de la red
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)

        x = F.dropout(x, training= self.training)
        x = self.fc2(x)

        x = F.log_softmax(x)
        return x