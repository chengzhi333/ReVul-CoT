# coding:utf-8
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

# 读取预测结果
df = pd.read_excel("RQ2/9-1/test 9-1.xlsx")
# df = pd.read_excel("test_all_predicted5.xlsx")
# 过滤掉 Predicted 为空的行
df_non_empty = df[df["Predicted"].notna() & (df["Predicted"] != "")].copy()

# 转换为大写，避免大小写影响
df_non_empty["Predicted"] = df_non_empty["Predicted"].str.upper().str.strip()
df_non_empty["Base Severity"] = df_non_empty["Base Severity"].str.upper().str.strip()

if df_non_empty.empty:
    print("没有有效预测结果，无法计算指标。")
else:
    actual = df_non_empty["Base Severity"].tolist()
    predicted = df_non_empty["Predicted"].tolist()

    # 1. Accuracy
    acc = accuracy_score(actual, predicted)

    # 2. Precision, Recall, F1 (macro average)
    precision_ma, recall_ma, f1_ma, _ = precision_recall_fscore_support(
        actual, predicted, average='macro'
    )

    # 3. MCC
    mcc = matthews_corrcoef(actual, predicted)

    print(f"有效预测数量: {len(actual)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Macro): {precision_ma:.4f}")
    print(f"Recall (Macro): {recall_ma:.4f}")
    print(f"F1-score (Macro): {f1_ma:.4f}")
    print(f"MCC: {mcc:.4f}")
