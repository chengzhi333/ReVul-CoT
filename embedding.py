# coding:utf-8
import psycopg2
import json
import time
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 256  # 每条文本/代码最大token长度
POOLING = 'first_last_avg'  # 池化方式

# ================== 数据库连接 ===================
conn = psycopg2.connect(
    dbname="rag-vul",
    user="postgres",
    password="123456",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# ================== 嵌入模型 ===================
# CodeBERT
code_model_name = "microsoft/codebert-base"
code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
code_model = AutoModel.from_pretrained(code_model_name)
code_model.to(DEVICE)
code_model.eval()

# text2vec-base-multilingual
desc_model_name = "shibing624/text2vec-base-multilingual"
desc_tokenizer = AutoTokenizer.from_pretrained(desc_model_name)
desc_model = AutoModel.from_pretrained(desc_model_name)
desc_model.to(DEVICE)
desc_model.eval()


# ================== 向量函数 ===================
def embed_code(text):
    inputs = code_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = code_model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if POOLING == 'first_last_avg':
            vec = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif POOLING == 'last_avg':
            vec = hidden_states[-1].mean(dim=1)
        elif POOLING == 'last2avg':
            vec = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        else:
            raise Exception(f"Unknown pooling {POOLING}")
    vec = vec.cpu().numpy()[0]
    vec = vec / np.linalg.norm(vec)  # 标准化
    return vec.tolist()


def embed_desc(text):
    inputs = desc_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = desc_model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if POOLING == 'first_last_avg':
            vec = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif POOLING == 'last_avg':
            vec = hidden_states[-1].mean(dim=1)
        elif POOLING == 'last2avg':
            vec = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        else:
            raise Exception(f"Unknown pooling {POOLING}")
    vec = vec.cpu().numpy()[0]
    vec = vec / np.linalg.norm(vec)  # 标准化
    return vec.tolist()


# ================== 主程序 ===================
def main():
    df = pd.read_excel("knowledge/train_all_with_nvd_cwe.xlsx")
    ids_map = {}  # FAISS idx -> 数据库 ID 映射

    code_vectors = []
    desc_vectors = []

    for faiss_idx, row in df.iterrows():
        cve_id = row.get("cve_id", "")
        code = row.get("func_before", "")
        desc = row.get("description", "")
        base_score = row.get("Base Score", None)
        base_severity = row.get("Base Severity", None)

        # --- 处理 cwe_ids ---
        cwe_ids_raw = row.get("cwe_ids", "")
        if pd.isna(cwe_ids_raw) or not str(cwe_ids_raw).strip():
            cwe_ids = []
        elif isinstance(cwe_ids_raw, str):
            cwe_ids = [cid.strip() for cid in cwe_ids_raw.split(",") if cid.strip()]
        elif isinstance(cwe_ids_raw, list):
            cwe_ids = cwe_ids_raw
        else:
            cwe_ids = []

        # --- 处理 nvd_info ---
        nvd_info = row.get("nvd_info", "{}")
        if pd.isna(nvd_info) or not str(nvd_info).strip():
            nvd_info = "{}"
        if not isinstance(nvd_info, str):
            nvd_info = json.dumps(nvd_info, ensure_ascii=False)

        # --- 处理 cwe_info ---
        cwe_info = row.get("cwe_info", "[]")
        if pd.isna(cwe_info) or not str(cwe_info).strip():
            cwe_info = "[]"
        if not isinstance(cwe_info, str):
            cwe_info = json.dumps(cwe_info, ensure_ascii=False)

        print(f"[INFO] 处理 {faiss_idx + 1}/{len(df)}: {cve_id}")

        # 向量化
        code_emb = embed_code(code)
        desc_emb = embed_desc(desc)

        code_vectors.append(code_emb)
        desc_vectors.append(desc_emb)

        # 插入数据库
        cur.execute("""
                    INSERT INTO vulnerabilities
                    (cve_id, cwe_ids, code, description, base_score, base_severity, nvd_info, cwe_info, code_embedding,
                     desc_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """, (
                        cve_id,
                        cwe_ids,
                        code,
                        desc,
                        base_score,
                        base_severity,
                        nvd_info,
                        cwe_info,
                        code_emb,
                        desc_emb
                    ))
        vuln_id = cur.fetchone()[0]
        conn.commit()

        # 保存 FAISS idx -> 数据库 id
        ids_map[faiss_idx] = vuln_id

        # 构建 FAISS 索引
    print("[INFO] 构建 FAISS 索引...")
    code_vectors_np = np.array(code_vectors, dtype='float32')
    desc_vectors_np = np.array(desc_vectors, dtype='float32')

    index_code = faiss.IndexFlatL2(code_vectors_np.shape[1])
    index_code.add(code_vectors_np)

    index_desc = faiss.IndexFlatL2(desc_vectors_np.shape[1])
    index_desc.add(desc_vectors_np)

    # 保存 FAISS 索引
    faiss.write_index(index_code, "faiss/faiss_index_code.index")
    faiss.write_index(index_desc, "faiss/faiss_index_desc.index")
    print("[INFO] FAISS 索引已保存！")

    # 保存 id_map.json
    with open("faiss/id_map.json", "w", encoding="utf-8") as f:
        json.dump(ids_map, f, indent=2, ensure_ascii=False)

    print("✅ 所有漏洞写入完成！FAISS idx -> 数据库 ID 已保存到 id_map.json")

if __name__ == "__main__":
    main()
    cur.close()
    conn.close()
