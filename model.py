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
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
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

class ResNet(nn.Module):
    def __init__(self, in_channels, nb_blocks, nb_out):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, 16, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(16)

        layer1 = []
        for _ in range(nb_blocks):
            layer1.append(ResBlock(16, 16))
        self.layer1 = nn.Sequential(*layer1)

        layer2 = [nn.MaxPool2d(2), ResBlock(16, 32)]
        for _ in range(nb_blocks - 1):
            layer2.append(ResBlock(32, 32))
        self.layer2 = nn.Sequential(*layer2)

        layer3 = [nn.MaxPool2d(2), ResBlock(32, 64)]
        for _ in range(nb_blocks - 1):
            layer3.append(ResBlock(64, 64))
        self.layer3 = nn.Sequential(*layer3)

        self.linear(64, nb_out)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        return x

# TODO: Transformer Encoder / Decoder Layer with ReZero

if __name__ == "__main__":
    x = torch.randn(1, 8)
    l = RZLinear(8, 8)
    y = l(x)

    print(y)
    print(y.shape)
