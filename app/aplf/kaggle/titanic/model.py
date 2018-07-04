import torch.nn as nn
import torch.nn.functional as F


class TitanicNet(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.fc0 = nn.Linear(input_len, int(input_len/2))
        self.fc1 = nn.Linear(int(input_len/2), int(input_len/3))
        self.fc2 = nn.Linear(int(input_len/3), int(input_len/4))
        self.fc3 = nn.Linear(int(input_len/4), int(input_len/5))
        self.fc4 = nn.Linear(int(input_len/5), output_len)

    def forward(self, x):
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x
