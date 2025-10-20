# coding:utf-8
import pandas as pd
import re
import requests
import psycopg2
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import transformers
import os
import json
from openai import OpenAI
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================== 配置 ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 256
POOLING = 'first_last_avg'
ALPHA = 0.6  # code 权重
BETA = 0.4  # description 权重
TOPK = 5

# ================== 数据库连接 ===================
conn = psycopg2.connect(
    dbname="rag-vul",
    user="postgres",
    password="123456",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# ================== 加载 FAISS 索引 ==================
index_code = faiss.read_index("faiss/faiss_index_code.index")
index_desc = faiss.read_index("faiss/faiss_index_desc.index")

# ================== FAISS idx → 数据库 id 映射 ==================
with open("faiss/id_map.json", "r", encoding="utf-8") as f:
    id_map = json.load(f)  # id_map: {faiss_idx_str: db_id}

def get_vuln_info_by_faiss_idx(idx):
    db_id = id_map.get(str(idx))  # FAISS 索引对应数据库 id
    if db_id is None:
        return None
    cur.execute("""
                SELECT cve_id,
                       cwe_ids,
                       code,
                       description,
                       base_score,
                       base_severity,
                       nvd_info,
                       cwe_info
                FROM vulnerabilities
                WHERE id = %s
                """, (db_id,))
    row = cur.fetchone()
    if row:
        return {
            "cve_id": row[0],
            "cwe_ids": row[1],
            "code": row[2],
            "description": row[3],
            "base_score": row[4],
            "base_severity": row[5],
            "nvd_info": row[6],
            "cwe_info": row[7]
        }
    return None

# ================== 加载模型 ==================
code_model_name = "microsoft/codebert-base"
desc_model_name = "shibing624/text2vec-base-multilingual"

code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
code_model = AutoModel.from_pretrained(code_model_name).to(DEVICE)
code_model.eval()

desc_tokenizer = AutoTokenizer.from_pretrained(desc_model_name)
desc_model = AutoModel.from_pretrained(desc_model_name).to(DEVICE)
desc_model.eval()

# ================== 向量化函数 ==================
def embed_text(text, tokenizer, model, max_length=MAX_LENGTH, pooling=POOLING):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if pooling == 'first_last_avg':
            vec = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif pooling == 'last_avg':
            vec = hidden_states[-1].mean(dim=1)
        elif pooling == 'last2avg':
            vec = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
    vec = vec.cpu().numpy()[0]
    return vec / np.linalg.norm(vec)

def embed_code(text):
    return embed_text(text, code_tokenizer, code_model)

def embed_desc(text):
    return embed_text(text, desc_tokenizer, desc_model)

# ================== 多模态 RAG 检索 ==================
def rag_multimodal_search(query_code, query_desc, topk=TOPK, alpha=ALPHA, beta=BETA):
    # 1. 获取向量
    code_vec = np.array(embed_code(query_code), dtype='float32').reshape(1, -1)
    desc_vec = np.array(embed_desc(query_desc), dtype='float32').reshape(1, -1)

    # 2. L2搜索，取各自 topk
    _, idx_code = index_code.search(code_vec, topk * 2)
    _, idx_desc = index_desc.search(desc_vec, topk * 2)

    # 3. 合并候选索引
    candidate_idx = list(set(idx_code[0].tolist() + idx_desc[0].tolist()))

    results = []
    for idx in candidate_idx:
        db_code_vec = index_code.reconstruct(idx)
        db_desc_vec = index_desc.reconstruct(idx)

        # 4. 计算余弦相似度
        code_sim = np.dot(code_vec, db_code_vec).item()
        desc_sim = np.dot(desc_vec, db_desc_vec).item()

        # 5. 加权
        score = alpha * code_sim + beta * desc_sim

        vuln_info = get_vuln_info_by_faiss_idx(idx)
        if vuln_info:
            vuln_info["score"] = score
            results.append(vuln_info)

    # 设置RAG检索个数（默认5个）
    k = topk - 2

    # 按分数排序 topK
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
    for item in results:
        print(f"Score: {item['score']:.4f}, CVE ID: {item['cve_id']}, CWE IDs: {item['cwe_ids']}, Base Score: {item['base_score']}, "
              f"Base Severity: {item['base_severity']}, Code: {item['code']}, Description: {item['description']}, "
              f"NVD Info: {item['nvd_info']}, CWE Info: {item['cwe_info']}")

    return results

# ================== DeepSeek 调用 ==================
client = OpenAI(
    api_key="sk-2b62d88b75f041c29aa844889daadfc5",
    base_url="https://api.deepseek.com"
)

# ================== 少样本COT ==================
def predict_vuln_level_fewshot_cot(query_code, query_desc, topk_samples):
    """
    少样本COT 先分析相似漏洞样本，在结合分析直接生成目标漏洞的结构化解释性知识，最后生成漏洞等级
    """
    prompt = "Your task is to analyze vulnerabilities step by step and finally output only the severity of the target vulnerability.\n\n"

    # Step1: 分析示例
    prompt += "Step 1: Analyze the following several similar vulnerability samples. For each sample, consider:\n"
    prompt += "- Functional semantics of the code\n"
    prompt += "- Vulnerability causes\n"
    prompt += "- Fixing solutions\n"
    prompt += "- Impact scope (affected modules, attack surface)\n"
    prompt += "- Exploitability (attack vector, authentication, preconditions)\n"
    prompt += "- Impact type (confidentiality, integrity, availability, privilege escalation, RCE, data leak)\n"
    prompt += "- Security context (required privileges, privilege level gained)\n"
    prompt += "- Severity mapping clues (why it was classified as LOW, MEDIUM, HIGH, or CRITICAL)\n"
    prompt += "- Official severity (Base Severity)\n\n"

    for i, item in enumerate(topk_samples):
        prompt += f"Sample {i + 1}:\n"
        prompt += f"- CVE ID: {item['cve_id']}\n"
        prompt += f"- CWE IDs: {item['cwe_ids']}\n"
        prompt += f"- Base Score: {item['base_score']}\n"
        prompt += f"- Base Severity: {item['base_severity']}\n"
        prompt += f"- Code: {item['code']}\n"
        prompt += f"- Description: {item['description']}\n"
        prompt += f"- NVD Info: {item['nvd_info']}\n"
        prompt += f"- CWE Info: {item['cwe_info']}\n\n"
        # prompt += f"- Similarity Score: {item['score']}\n\n"

    # Step2: 分析目标漏洞
    prompt += "Step 2: Based on the patterns observed in Step 1, analyze the target vulnerability.\n"
    prompt += "Generate structured explanatory knowledge before deciding severity:\n"
    prompt += "Explanatory Knowledge:\n"
    prompt += "1. Functional Semantics: [...]\n"
    prompt += "2. Vulnerability Causes: [...]\n"
    prompt += "3. Fixing Solutions: [...]\n\n"
    prompt += "4. Impact Scope: [Affected components/modules, size of attack surface]\n"
    prompt += "5. Exploitability: [Attack vector, authentication required, preconditions]\n"
    prompt += "6. Impact Type: [Confidentiality, Integrity, Availability, privilege escalation, RCE, data leak]\n"
    prompt += "7. Security Context: [Required privileges for exploitation, privilege level gained]\n"
    prompt += "8. Severity Mapping Clues: [Summarize why similar cases were rated at certain severity levels]\n\n"

    prompt += "Target Vulnerability:\n"
    prompt += f"- Code: {query_code}\n"
    prompt += f"- Description: {query_desc}\n\n"

    # Step3: 输出严重等级
    prompt += "Step 3: Based on Step 1 and Step 2, You only need to output the severity level of the target vulnerability.\n"
    prompt += "Do not output any explanation, reasoning process, or the severity levels of the previous sample examples.\n"

    # 计算token
    chat_tokenizer_dir = "./deepseek_v3_tokenizer"  # 本地 tokenizer 路径
    tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=True)
    system_content = "You are an expert in code vulnerability assessment, and you will rate the vulnerabilities based on the following scoring criteria:\n0.1-3.9: LOW, 4.0-6.9: MEDIUM, 7.0-8.9: HIGH, 9.0-10.0: CRITICAL."
    user_content = prompt
    system_tokens = tokenizer.encode(system_content)
    user_tokens = tokenizer.encode(user_content)
    total_tokens = len(system_tokens) + len(user_tokens)
    print("Total tokens:", total_tokens)

    # response = client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[
    #         {"role": "system", "content": "You are an expert in code vulnerability assessment, and you will rate the vulnerabilities based on the following scoring criteria:\n0.1-3.9: LOW, 4.0-6.9: MEDIUM, 7.0-8.9: HIGH, 9.0-10.0: CRITICAL."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature = 0,
    #     stream=False
    # )
    #
    # level = response.choices[0].message.content
    # return level

    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-fvptcnilnjiaxfjmlukkpcohiannuozwgfaknxaffmycfugd",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-V3.1",
        "messages": [
            {"role": "system", "content": "You are an expert in code vulnerability assessment, and you will rate the vulnerabilities based on the following scoring criteria:\n0.1-3.9: LOW, 4.0-6.9: MEDIUM, 7.0-8.9: HIGH, 9.0-10.0: CRITICAL.."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    res = requests.post(url, headers=headers, json=payload)
    res_json = res.json()

    # 提取 content
    output = res_json['choices'][0]['message']['content']
    return output

# ================== 主调用函数 ==================
def predict_vuln_level_rag_llm(query_code, query_desc):
    # 1. RAG 多模态检索 topK 样本
    topk_samples = rag_multimodal_search(query_code, query_desc)

    # 2. COT
    level = predict_vuln_level_fewshot_cot(query_code, query_desc, topk_samples)
    return level

# ================== 运行 ==================
if __name__ == "__main__":
    """
    运行
    """
    # 读取 Excel 文件
    input_file = "datasets/test/test_all.xlsx"
    output_file = "test_all_predicted5.xlsx"
    temp_file = "test_all_predicted_temp5.xlsx"

    VALID_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    # 判断是否已经有预测文件
    if os.path.exists(output_file):
        df = pd.read_excel(output_file)
        print(f"继续运行：已加载 {output_file}")
    else:
        df = pd.read_excel(input_file)
        df["Predicted"] = ""  # 初始化预测列
        print(f"新运行：加载 {input_file}")

    # 找到 Predicted 不在有效等级集合的行
    rows_to_predict = df[~df["Predicted"].astype(str).str.strip().isin(VALID_LEVELS)].index

    if len(rows_to_predict) == 0:
        print("所有行都已经预测完成！")
    else:
        for idx in rows_to_predict:
            row = df.loc[idx]
            query_code = row['func_before']
            query_desc = row['description']

            try:
                # 预测漏洞等级
                level = predict_vuln_level_rag_llm(query_code, query_desc)
                print(f"Row {idx}: {level} (Base Severity: {row['Base Severity']})")
            except Exception as e:
                print(f"Error at row {idx}: {e}")
                level = ""

            # 写入预测结果
            df.at[idx, "Predicted"] = level

            # 保存到临时文件，再覆盖
            df.to_excel(temp_file, index=False)
            os.replace(temp_file, output_file)

        print(f"预测完成，结果已保存到 {output_file}")

