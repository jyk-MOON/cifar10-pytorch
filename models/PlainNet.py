import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # 批量归一化
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out

class PlainNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PlainNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # 输出: 32x32
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)   # 输出: 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 输出: 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 输出: 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 输出: 4x4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: 1x1
        self.fc = nn.Linear(512, num_classes)

    # 构建多个 PlainBlock 组成的一层
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(PlainBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels # 上一层的输出为这一层的输入
        for _ in range(1, blocks): # 从第二个block开始，每一个block的输入通道 = 输出通道
            layers.append(PlainBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
