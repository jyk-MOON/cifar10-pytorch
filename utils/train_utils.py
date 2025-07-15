import os
import torch
import torch.nn.functional as F
import csv

class Trainer:
    def __init__(self, model, device, optimizer, train_loader, test_loader, model_name='model', scheduler=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_name = model_name
        self.scheduler = scheduler

    def train_one_epoch(self):
        self.model.train()
        # 累计损失、正确预测数、样本总数
        total_loss = 0.0
        correct = 0
        total = 0

        # 创建保存日志的文件夹
        project_root = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
        self.log_dir = os.path.join(project_root, '..', 'training_logs')  # 上一级，即项目根目录
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f'{self.model_name}_training_log.csv')

        if not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device) # 把数据放到GPU上

            self.optimizer.zero_grad()
            outputs = self.model(images) # 前向传播
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)  # 计算当前批次的总损失
            _, predicted = outputs.max(1)  # outputs.max(1)返回最大值和索引（只需要索引）
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy # 返回平均损失和正确率

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()
            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch {epoch}/{epochs} — "
                  f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} — "
                  f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

            # 记录到CSV文件
            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.4f}",
                    f"{train_acc:.4f}",
                    f"{val_loss:.4f}",
                    f"{val_acc:.4f}"
                ])