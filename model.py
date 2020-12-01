import torch
import torch.nn as nn
import torch.nn.functional as F

class RZLinear(nn.Linear):
    def __init__(self, *args, start_alpha=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.tensor(start_alpha))

    def forward(self, x):
        return x + self.alpha * F.elu(super().forward(x))

# class RZConv2d(nn.Conv2d):
    # def __init__(self, *args, start_alpha=0.0, **kwargs):
        # super().__init__(*args, **kwargs)
        # self.alpha = nn.Parameter(torch.tensor(start_alpha))
        # self.bn = nn.BatchNorm2d(args[1])

    # def forward(self, x):
        # return x + self.alpha * F.elu(self.bn(super().forward(x)))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = self.conv2(self.conv1(x)) * self.alpha + x
        return out

# TODO: Transformer Encoder / Decoder Layer with ReZero

if __name__ == "__main__":
    x = torch.randn(1, 8)
    l = RZLinear(8, 8)
    y = l(x)

    print(y)
    print(y.shape)
