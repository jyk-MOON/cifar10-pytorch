import re
import csv

# 输入输出文件路径
input_path = r"C:\Users\86165\Desktop\Programs\ResNet_20.txt"
output_path = r"D:\pycharm_project\project2\training_logs\ResNet_20_training_log.csv"

# 正则表达式用于提取每一行中的数值
pattern = re.compile(
    r"Train loss: ([\d.]+), Train acc: ([\d.]+) — Val loss: ([\d.]+), Val acc: ([\d.]+)"
)

# 存储提取的数据
data = []

with open(input_path, "r", encoding="utf-8") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            data.append([float(match.group(1)), float(match.group(2)),
                         float(match.group(3)), float(match.group(4))])

# 写入CSV文件
with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Train loss", "Train acc", "Val loss", "Val acc"])  # 写表头
    writer.writerows(data)  # 写数据

print("转换完成，CSV已保存至：", output_path)
