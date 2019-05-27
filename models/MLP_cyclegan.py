import torch
import torch.nn as nn


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2048, 256)
        #self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()

        self.apply(weights_init)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = nn.Sigmoid()(out)

        return out


class MLPGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 2048)

        self.relu = nn.ReLU()

        #self.apply(weights_init)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = nn.Sigmoid()(out)

        return out







