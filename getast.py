# coding:utf-8
from tree_sitter import Language, Parser
import pandas as pd
import os
import sys

# 确保 Excel 文件存在
excel_file_path = "datasets/megavul_simple_cpp_all.xlsx"

# 读取数据
data = pd.read_excel(excel_file_path)  # 使用 read_excel 来读取 Excel 文件
code = data['func_before']  # 假设 'func_before' 列包含代码

# 加载编译后的 C++ 语言解析器
CPP_LANGUAGE = Language('build1/my-languages.so', 'cpp')
cpp_parser = Parser()
cpp_parser.set_language(CPP_LANGUAGE)

# 创建一个空列表来存储成功解析的行
successful_rows = []

# 解析代码并生成 AST
for index, code_str in enumerate(code):
    try:
        # 尝试解析代码并生成 AST
        tree = cpp_parser.parse(bytes(code_str, "utf8"))

        # 如果生成的 AST 不为空（有效）
        if tree is not None:
            # 将成功解析的行添加到 successful_rows 列表中
            successful_rows.append(data.iloc[index])  # data.iloc[index] 获取该行数据
    except Exception as e:
        # 如果解析过程中有任何异常，跳过该行
        print(f"Error parsing line {index}: {e}")

# 将成功解析的行转换为新的 DataFrame
successful_data = pd.DataFrame(successful_rows)

# 确保输出路径存在
output_dir = "datasets"

# 保存成功解析的行到 Excel 文件
successful_data.to_excel(os.path.join(output_dir, "megavul_simple_cpp_success_ast.xlsx"), index=False)

print(f"Successfully parsed {len(successful_data)} rows and saved to 'megavul_simple_cpp_success_ast.xlsx'.")
