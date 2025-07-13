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
    epochs = 100
    learning_rate = 0.001

    # 加载数据
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(128)

    # 初始化模型、优化器
    model = ResNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 实例化Trainer并开始训练
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        model_name='AlexNet'
    )
    trainer.train(epochs)

if __name__ == "__main__":
    main()
