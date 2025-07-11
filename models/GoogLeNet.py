import torch
import torch.nn as nn
import torch.nn.functional as F

# Inception模块定义
class Inception(nn.Module):
    # 参数表示：输入特征图的通道数、第一分支的输出通道数、第二分支中 1x1 卷积的输出通道数（用于降低维度）、第二分支的输出通道数
    # 第三分支中 1x1 卷积的输出通道数、第三分支的输出通道数、第四分支中 max pooling 后的输出通道数
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3,
                 ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)

# GoogLeNet主结构
#  每个Inception模块会把输入同时传入四个不同的路径（分支），这四个路径是并行计算的，之后再将它们的输出在通道维度拼接起来
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入：3x32x32的图片，将其变为64通道图片
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 输出: 16x16
        )

        self.inception_a = Inception(64, 32, 48, 64, 8, 16, 16)  # 输出通道数: 32+64+16+16=128
        self.inception_b = Inception(128, 64, 64, 96, 16, 32, 32)  # 输出: 64+96+32+32=224

        self.maxpool = nn.MaxPool2d(2, 2)  # 输出: 8x8

        self.inception_c = Inception(224, 64, 48, 64, 8, 16, 16)  # 输出: 64+64+16+16=160

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: 160x1x1
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(160, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.inception_a(x)
        x = self.inception_b(x)
        x = self.maxpool(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        return x
