# coding:utf-8
import os
import json
import psycopg2
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import transformers

# ================== 配置 ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 256
POOLING = 'first_last_avg'
ALPHA = 0.6  # code 权重
BETA  = 0.4  # description 权重
TOPK  = 5

# tokenizer（用于统计 token 数）
CHAT_TOKENIZER_DIR = "./deepseek_v3_tokenizer"

# ================== 数据库连接 ==================
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
    id_map = json.load(f)  # {faiss_idx_str: db_id}

def get_vuln_info_by_faiss_idx(idx: int):
    db_id = id_map.get(str(idx))
    if db_id is None:
        return None
    cur.execute("""
        SELECT cve_id, cwe_ids, code, description, base_score,
               base_severity, nvd_info, cwe_info
        FROM vulnerabilities
        WHERE id = %s
    """, (db_id,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "cve_id": row[0] or "",
        "cwe_ids": row[1] or "",
        "code": row[2] or "",
        "description": row[3] or "",
        "base_score": row[4],
        "base_severity": row[5] or "",
        "nvd_info": row[6] or "",
        "cwe_info": row[7] or ""
    }

# ================== 加载嵌入模型==================
code_model_name = "microsoft/codebert-base"
desc_model_name = "shibing624/text2vec-base-multilingual"

code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
code_model     = AutoModel.from_pretrained(code_model_name).to(DEVICE)
code_model.eval()

desc_tokenizer = AutoTokenizer.from_pretrained(desc_model_name)
desc_model     = AutoModel.from_pretrained(desc_model_name).to(DEVICE)
desc_model.eval()

def embed_text(text, tokenizer, model, max_length=MAX_LENGTH, pooling=POOLING):
    if not text:
        text = ""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states
        if pooling == 'first_last_avg':
            vec = (hs[-1] + hs[1]).mean(dim=1)
        elif pooling == 'last_avg':
            vec = hs[-1].mean(dim=1)
        elif pooling == 'last2avg':
            vec = (hs[-1] + hs[-2]).mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
    vec = vec.cpu().numpy()[0]
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec

def embed_code(text): return embed_text(text, code_tokenizer, code_model)
def embed_desc(text): return embed_text(text, desc_tokenizer, desc_model)

# ================== 多模态 RAG 检索（与实验一致） ==================
def rag_multimodal_search(query_code, query_desc, topk=TOPK, alpha=ALPHA, beta=BETA):
    code_vec = np.array(embed_code(query_code), dtype='float32').reshape(1, -1)
    desc_vec = np.array(embed_desc(query_desc), dtype='float32').reshape(1, -1)

    # L2 search
    _, idx_code = index_code.search(code_vec, topk * 2)
    _, idx_desc = index_desc.search(desc_vec, topk * 2)

    candidate_idx = list(set(idx_code[0].tolist() + idx_desc[0].tolist()))
    results = []
    for idx in candidate_idx:
        db_code_vec = index_code.reconstruct(idx)
        db_desc_vec = index_desc.reconstruct(idx)

        code_sim = np.dot(code_vec, db_code_vec).item()
        desc_sim = np.dot(desc_vec, db_desc_vec).item()
        score = alpha * code_sim + beta * desc_sim

        vuln = get_vuln_info_by_faiss_idx(idx)
        if vuln:
            vuln["score"] = score
            results.append(vuln)

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:topk]

    for item in results:
        print(f"Score: {item['score']:.4f}, CVE ID: {item['cve_id']}, CWE IDs: {item['cwe_ids']}, Base Score: {item['base_score']}, "
              f"Base Severity: {item['base_severity']}, Code: {item['code']}, Description: {item['description']}, "
              f"NVD Info: {item['nvd_info']}, CWE Info: {item['cwe_info']}")
    print('--------------------------------------------------------------------------------------------------------------------------')
    return results

# ================== Prompt 模板（system + user） ==================
SYSTEM_CONTENT = (
    "You are an expert in code vulnerability assessment, and you will rate the "
    "vulnerabilities based on the following scoring criteria:\n"
    "0.1-3.9: LOW, 4.0-6.9: MEDIUM, 7.0-8.9: HIGH, 9.0-10.0: CRITICAL."
)

def build_fewshot_cot_prompt(query_code, query_desc, topk_samples, tokenizer):
    """
    构造 few-shot CoT 提示词；同时统计每个检索样本的 token 数量
    返回: (full_prompt, per_sample_token_list)
    """
    prompt = "Your task is to analyze vulnerabilities step by step and finally output only the severity of the target vulnerability.\n\n"

    # Step 1 说明
    head = (
        "Step 1: Analyze the following several similar vulnerability samples. For each sample, consider:\n"
        "- Functional semantics of the code\n"
        "- Vulnerability causes\n"
        "- Fixing solutions\n"
        "- Impact scope (affected modules, attack surface)\n"
        "- Exploitability (attack vector, authentication, preconditions)\n"
        "- Impact type (confidentiality, integrity, availability, privilege escalation, RCE, data leak)\n"
        "- Security context (required privileges, privilege level gained)\n"
        "- Severity mapping clues (why it was classified as LOW, MEDIUM, HIGH, or CRITICAL)\n"
        "- Official severity (Base Severity)\n\n"
    )
    prompt += head

    per_sample_tokens = []
    # 逐样本正文 + 逐样本 token
    for i, item in enumerate(topk_samples):
        sample_text = (
            f"Sample {i + 1}:\n"
            f"- CVE ID: {item['cve_id']}\n"
            f"- CWE IDs: {item['cwe_ids']}\n"
            f"- Base Score: {item['base_score']}\n"
            f"- Base Severity: {item['base_severity']}\n"
            f"- Code: {item['code']}\n"
            f"- Description: {item['description']}\n"
            f"- NVD Info: {item['nvd_info']}\n"
            f"- CWE Info: {item['cwe_info']}\n\n"
        )
        prompt += sample_text
        per_sample_tokens.append(len(tokenizer.encode(sample_text)))

    # Step 2 + 目标漏洞
    body2 = (
        "Step 2: Based on the patterns observed in Step 1, analyze the target vulnerability.\n"
        "Generate structured explanatory knowledge before deciding severity:\n"
        "Explanatory Knowledge:\n"
        "1. Functional Semantics: [...]\n"
        "2. Vulnerability Causes: [...]\n"
        "3. Fixing Solutions: [...]\n\n"
        "4. Impact Scope: [Affected components/modules, size of attack surface]\n"
        "5. Exploitability: [Attack vector, authentication required, preconditions]\n"
        "6. Impact Type: [Confidentiality, Integrity, Availability, privilege escalation, RCE, data leak]\n"
        "7. Security Context: [Required privileges for exploitation, privilege level gained]\n"
        "8. Severity Mapping Clues: [Summarize why similar cases were rated at certain severity levels]\n\n"
        "Target Vulnerability:\n"
        f"- Code: {query_code}\n"
        f"- Description: {query_desc}\n\n"
        "Step 3: Based on Step 1 and Step 2, You only need to output the severity level of the target vulnerability.\n"
        "Do not output any explanation, reasoning process, or the severity levels of the previous sample examples.\n"
    )
    prompt += body2
    return prompt, per_sample_tokens

# ================== 统计函数：返回 token 数 ==================
def count_tokens(system_content: str, user_content: str, tokenizer):
    sys_tokens  = len(tokenizer.encode(system_content))
    user_tokens = len(tokenizer.encode(user_content))
    return sys_tokens, user_tokens, sys_tokens + user_tokens

# ================== 主流程：遍历测试集，统计 token 用量 ==================
def main():
    # 读取测试集
    input_file = "datasets/test/test_all.xlsx"
    output_file = "token usages/token_usage_stats.csv"

    df = pd.read_excel(input_file)
    assert "func_before" in df.columns and "description" in df.columns, "缺少 func_before 或 description 列"

    # 使用 transformers 的方法来计算 token
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CHAT_TOKENIZER_DIR,
        local_files_only=True,
        trust_remote_code=False
    )

    rows = []
    for idx, row in df.iterrows():
        query_code = str(row["func_before"]) if not pd.isna(row["func_before"]) else ""
        query_desc = str(row["description"]) if not pd.isna(row["description"]) else ""

        # RAG 检索 topK 样本
        topk_samples = rag_multimodal_search(query_code, query_desc, topk=TOPK, alpha=ALPHA, beta=BETA)

        # 构造 few-shot CoT prompt + 逐样本 token
        full_prompt, per_sample_tokens = build_fewshot_cot_prompt(query_code, query_desc, topk_samples, tokenizer)

        # 统计 token（system + user）
        sys_toks, user_toks, total_toks = count_tokens(SYSTEM_CONTENT, full_prompt, tokenizer)

        # 样本 token 汇总和占比
        sample_total_toks = int(np.sum(per_sample_tokens)) if per_sample_tokens else 0
        sample_avg_toks   = float(np.mean(per_sample_tokens)) if per_sample_tokens else 0.0
        sample_max_toks   = int(np.max(per_sample_tokens)) if per_sample_tokens else 0
        sample_share_user = (sample_total_toks / user_toks) if user_toks > 0 else 0.0

        row_out = {
            "row_id": idx,
            "system_tokens": sys_toks,
            "user_tokens": user_toks,
            "total_tokens": total_toks,
            "sample_total_tokens": sample_total_toks,
            "sample_avg_tokens": round(sample_avg_toks, 2),
            "sample_max_tokens": sample_max_toks,
            "sample_token_share_user": round(sample_share_user, 6)
        }
        # 逐样本 token 明细
        for i, t in enumerate(per_sample_tokens, start=1):
            row_out[f"Sample{i}_tokens"] = int(t)

        rows.append(row_out)

        if (idx + 1) % 100 == 0:
            print(f"[Progress] processed {idx + 1} rows...")

    usage_df = pd.DataFrame(rows)
    usage_df.to_csv(output_file, index=False)
    print(f"[Done] per-row token usage saved to {output_file}")

    # 统计汇总
    def q(x, p): return float(x.quantile(p)) if len(x) else 0.0

    stats = {
        "N": len(usage_df),
        "total_min": int(usage_df["total_tokens"].min()),
        "total_max": int(usage_df["total_tokens"].max()),
        "total_mean": float(usage_df["total_tokens"].mean()),
        "total_median": float(usage_df["total_tokens"].median()),
        "total_std": float(usage_df["total_tokens"].std(ddof=1)),
        "p10": q(usage_df["total_tokens"], 0.10),
        "p25": q(usage_df["total_tokens"], 0.25),
        "p75": q(usage_df["total_tokens"], 0.75),
        "p90": q(usage_df["total_tokens"], 0.90),
        # 样本 token 汇总
        "sample_total_mean": float(usage_df["sample_total_tokens"].mean()),
        "sample_total_median": float(usage_df["sample_total_tokens"].median()),
        "sample_total_min": int(usage_df["sample_total_tokens"].min()),
        "sample_total_max": int(usage_df["sample_total_tokens"].max()),
        "sample_share_user_mean": float(usage_df["sample_token_share_user"].mean())
    }

    print("\n========== Token Usage Summary (Few-shot CoT with RAG) ==========")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    pd.DataFrame([stats]).to_csv("token usages/token_usage_summary.csv", index=False)
    print("[Done] summary saved to token_usage_summary.csv")


if __name__ == "__main__":
    main()
