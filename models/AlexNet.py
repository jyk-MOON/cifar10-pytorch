import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential( # 提取视觉特征，为后续分类做准备
            # CIFAR-10图像32x32较小，kernel、stride、padding做了改动
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # 输出: 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 16x16

            nn.Conv2d(64, 192, kernel_size=3, padding=1), # 输出: 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 8x8

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 输出: 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 输出: 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 输出: 8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 4x4
        )
        self.classifier = nn.Sequential( # 分类器部分，将特征映射转换成类别预测概率
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
