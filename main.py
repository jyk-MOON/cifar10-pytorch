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

    # 初始化模型、优化器
    model = ResNet(20, num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
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
