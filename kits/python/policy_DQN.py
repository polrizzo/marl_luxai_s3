import random
import torch.nn as nn
from collections import deque


class DQN(nn.Module):
    # def __init__(self, input_size, hidden_size, output_size):
    def __init__(self, channels, hidden_size, output_size, dropout=0.1):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            # 1. Conv block (output 16x24x24)
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            # 2. Conv block (output 64x24x24)
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # 3. Conv block (output 64x24x24)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # MaxPool (output 64x12x12)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4. Conv block (output 128x12x12)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # MaxPool (output 128x6x6)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5. Conv block (output 128x6x6)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # MaxPool (output 128x3x3)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Flatten (output 128*3*3)
            nn.Flatten(),
            nn.Dropout(dropout),
            # FC Layer (output hidden_size)
            nn.Linear(in_features=128*3*3, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            # FC Layer (output hidden_size)
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            # FC Layer (output 6)
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)