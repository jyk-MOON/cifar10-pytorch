import os
import pandas as pd
import matplotlib.pyplot as plt

# 获取相对路径（以 viewer 文件为基准）
current_dir = os.path.dirname(__file__)
log_dir = os.path.join(current_dir, '..', 'training_logs')

# 模型与路径映射
log_files = {
    "AlexNet": os.path.join(log_dir, "AlexNet_training_log.csv"),
    "GoogLeNet": os.path.join(log_dir, "GoogLeNet_training_log.csv"),
    "PlainNet": os.path.join(log_dir, "PlainNet_training_log.csv"),
    "ResNet": os.path.join(log_dir, "ResNet_training_log.csv")
}

# 加载所有模型的数据
all_logs = {
    model: pd.read_csv(path)
    for model, path in log_files.items()
}

# 指标列表
metrics = ["Train loss", "Val loss", "Train acc", "Val acc"]

# 为每个指标绘制单独的图像
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for model, df in all_logs.items():
        plt.plot(df[metric], label=model)
    plt.title(metric, fontsize=16)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
