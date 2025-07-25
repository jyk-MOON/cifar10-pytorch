import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # 批量归一化，加快收敛
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None # 下采样分支
        if stride != 1 or in_channels != out_channels: # 如果步长不为1，或输入输出通道不同，做1×1卷积调整通道数
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    # 整个模块“学习 F(x)，使得 F(x) + x 能逼近目标输出”
    # F(x) 不是人工指定学什么，而是网络通过反向传播和优化过程自动学到的；这是一种“结构上的引导”，而非“内容上的指令”
    def forward(self, x):
        identity = x # 保存输入的x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, n, num_classes=10):
        super(ResNet, self).__init__()
        assert (6 * n + 2) in [20, 32, 44, 56, 110], "depth must be 6n+2"

        self.in_channels = 16  # 初始通道设为16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, # 输出：16x32x32
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # n = 3, 5, 7, 9, 18
        # 主层的深度为 3 * n
        self.layer1 = self._make_layer(16, n, stride=1)   # 32x32
        self.layer2 = self._make_layer(32, n, stride=2)   # 16x16
        self.layer3 = self._make_layer(64, n, stride=2)   # 8x8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 64x1
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels # 上一层的输出为这一层的输入
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

