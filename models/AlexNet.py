import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential( # 特征提取层，为后续分类做准备
            # CIFAR-10图像32x32较小，kernel、stride、padding做了改动
            # 网络会自动学习64个不同的过滤器，每个过滤器去“观察”和“理解”不同的图像结构
            # 不是我们告诉网络怎么提取特征，而是它自己试出来的；卷积核的初始化不同、梯度更新不同、训练目标要求通道“互补”，最终不同的卷积核学会关注边缘、颜色、纹理...
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # 输出: 64x32x32，输入3通道，输出64通道
            nn.ReLU(inplace=True), # inplace=True表示直接在原输入的内存上修改数据，节省内存
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 16x16

            nn.Conv2d(64, 192, kernel_size=3, padding=1), # 输出: 192x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 8x8

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 输出: 384x8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 输出: 256x8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 输出: 256x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 256x4x4
        )
        self.classifier = nn.Sequential( # 分类器部分，将特征映射转换成类别预测概率
            nn.Dropout(), # 暂退法，训练时随机丢弃某些神经元，防止过拟合
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # 展平
        x = self.classifier(x)
        return x
