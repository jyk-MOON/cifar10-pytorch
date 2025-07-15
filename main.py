import torch
from models.AlexNet import AlexNet
from models.GoogLeNet import GoogLeNet
from models.PlainNet import PlainNet
from models.ResNet import ResNet
from utils.train_utils import Trainer
from datasets.cifar10 import get_dataloaders

def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 164
    learning_rate = 0.001

    # 加载数据
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(128)

    # 初始化模型
    model = ResNet(5, num_classes=10).to(device)
    # 随机梯度下降优化器，momentum为动量项，weight_decay为权重衰减
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # 学习率调度器，在epoch数达到时，将当前学习率乘gamma
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[82, 123], gamma=0.1)

    # 实例化Trainer并开始训练
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        model_name='ResNet',
        scheduler=scheduler
    )
    trainer.train(epochs)

if __name__ == "__main__":
    main()
