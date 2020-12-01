import torch
import torch.nn as nn
import torch.nn.functional as F

class RZLinear(nn.Linear):
    def __init__(self, *args, start_alpha=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.tensor(start_alpha))

    def forward(self, x):
        return x + self.alpha * F.elu(super().forward(x))

class RZConv2d(nn.Conv2d):
    def __init__(self, *args, start_alpha=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.tensor(start_alpha))

    def forward(self, x):
        return x + self.alpha * F.elu(super().forward(x))

# TODO: Transformer Encoder / Decoder Layer with ReZero

if __name__ == "__main__":
    x = torch.randn(1, 8)
    l = RZLinear(8, 8)
    y = l(x)

    print(y)
    print(y.shape)

    x = torch.randn(1, 1, 8, 8)
    l = RZConv2d(1, 8, 3, padding=1)
    y = l(x)

    print(y)
    print(y.shape)
