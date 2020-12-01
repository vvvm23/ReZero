import torch
import torch.nn as nn
import torch.nn.functional as F

class RZLinear(nn.Linear):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class RZConv2d(nn.Conv2d):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class RZMultiHeadAttention(nn.MultiheadAttention):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    print("Aloha, World!")
