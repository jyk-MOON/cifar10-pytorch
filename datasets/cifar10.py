import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 配置参数
NUM_WORKERS = 2
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
USE_CUDA = torch.cuda.is_available()

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 随机以50%的概率对图像进行水平翻转
    transforms.RandomCrop(32, padding=4), # 数据增强，先在四周填充4个像素（32x32 → 40x40），然后随机裁剪出32x32的区域作为最终图像
    transforms.ToTensor(), # 把原始图像转换为Tensor 格式（C x H x W）；同时像素值缩放到 0~1 之间（除以 255）
    transforms.Normalize((0.4914, 0.4822, 0.4465), # 对图像的每个通道（R G B）进行标准化处理
                         (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# 下载数据集
def get_dataloaders(batch_size=128):
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA
    )
    return train_loader, test_loader, train_dataset, test_dataset

if __name__ == '__main__':
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(128)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    images, labels = next(iter(train_loader))
    print(f"一个 batch 的图像形状: {images.shape}")  # torch.Size([128, 3, 32, 32])，批次大小x通道x高x宽
    print(f"一个 batch 的标签形状: {labels.shape}")  # torch.Size([128])，这个批次中每张图像对应的标签，共10个类别（0~9）