# coding:utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv("datasets/megavul_simple_cpp_success_getast.csv")

# 统计不同 Base Severity 的数量
severity_counts = data['Base Severity'].value_counts()

# 确定每个分组的数量比例
train_proportions = severity_counts.apply(lambda x: int(x * 0.8))  # 80% 用于训练集
valid_proportions = severity_counts.apply(lambda x: int(x * 0.1))  # 10% 用于验证集
test_proportions = severity_counts - train_proportions - valid_proportions  # 剩余 10% 用于测试集

# 打印每个类别的数量比例
print("Training Set Proportions:", train_proportions)
print("Validation Set Proportions:", valid_proportions)
print("Test Set Proportions:", test_proportions)

# 使用 stratify 保证类别的比例均衡
train_samples = train_proportions.sum()  # 计算训练集样本总数
valid_samples = valid_proportions.sum()  # 计算验证集样本总数

# 划分训练集和剩余数据集
train_data, remaining_data = train_test_split(
    data, train_size=train_samples, stratify=data['Base Severity'], random_state=42
)

# 划分验证集和测试集
valid_data, test_data = train_test_split(
    remaining_data, train_size=valid_samples, stratify=remaining_data['Base Severity'], random_state=42
)

# 保存划分后的数据集到 Excel 文件
train_data.to_excel("D:/python project/pythonProject/RAG-LLM/datasets/train/train_all.xlsx", index=False)
valid_data.to_excel("D:/python project/pythonProject/RAG-LLM/datasets/valid/valid_all.xlsx", index=False)
test_data.to_excel("D:/python project/pythonProject/RAG-LLM/datasets/test/test_all.xlsx", index=False)

print("Data splitting is complete. The datasets are saved to the specified directories.")
