# coding:utf-8
import json
import time
import requests
import certifi
import urllib3
import pandas as pd
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================== NVD / CWE 配置 ===================
API_KEY = "b1e4a307-35ce-462b-bf6b-268680dd176e"
NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0/"
CWE_BASE_URL = "https://cwe.mitre.org/data/definitions/"
HEADERS = {"apiKey": API_KEY, "User-Agent": "Mozilla/5.0"}


# ================== 爬虫函数 ===================
def fetch_nvd_info(cve_id):
    try:
        params = {"cveId": cve_id}
        resp = requests.get(NVD_BASE_URL, headers=HEADERS, params=params, verify=certifi.where())
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return None
        cve = vulns[0].get("cve", {})
        metrics_all = cve.get("metrics", {})
        cvss_metrics = metrics_all.get("cvssMetricV31") or metrics_all.get("cvssMetricV30") or []
        cvss_v3x = {}
        if cvss_metrics:
            cvss_data = cvss_metrics[0].get("cvssData", {})
            cvss_v3x = {
                "cvssData": {
                    "vectorString": cvss_data.get("vectorString"),
                    "attackVector": cvss_data.get("attackVector"),
                    "attackComplexity": cvss_data.get("attackComplexity"),
                    "privilegesRequired": cvss_data.get("privilegesRequired"),
                    "userInteraction": cvss_data.get("userInteraction"),
                    "scope": cvss_data.get("scope"),
                    "confidentialityImpact": cvss_data.get("confidentialityImpact"),
                    "integrityImpact": cvss_data.get("integrityImpact"),
                    "availabilityImpact": cvss_data.get("availabilityImpact"),
                },
                "exploitabilityScore": cvss_metrics[0].get("exploitabilityScore"),
                "impactScore": cvss_metrics[0].get("impactScore"),
            }
        weaknesses = cve.get("weaknesses", [])
        cwe_ids = []
        for w in weaknesses:
            descs = w.get("description", [])
            for desc in descs:
                val = desc.get("value")
                if val and val.startswith("CWE-"):
                    cwe_ids.append(val)

        affected_cpes = []
        configs = cve.get('configurations', [])
        for config in configs:
            nodes = config.get('nodes', [])
            for node in nodes:
                cpe_matches = node.get('cpeMatch', [])
                for cpe in cpe_matches:
                    crit = cpe.get('criteria')
                    if crit:
                        affected_cpes.append(crit)

        return {
            "cve_id": cve_id,
            "cvss_v3x": cvss_v3x,
            "cwe_ids": list(set(cwe_ids)),
            "affected_cpes": affected_cpes
        }
    except Exception as e:
        print(f"[ERROR] 获取 NVD 数据失败 {cve_id}: {e}")
        return None


def fetch_cwe_info(cwe_id):
    url = f"{CWE_BASE_URL}{cwe_id.split('-')[1]}.html"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 获取标题
        title_tag = soup.find("h2")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 判断是否为 PROHIBITED
        prohibited = False
        a_tags = soup.find_all("a", href=True)
        for a in a_tags:
            if a.text.strip() == "Vulnerability Mapping":
                tool_span = a.find_next("span", class_="tool")
                if tool_span and "PROHIBITED" in tool_span.text:
                    prohibited = True
                    break

        # 若为 PROHIBITED 页面，仅提取 Summary
        if prohibited:
            summary_text = ""
            summary_block = soup.find("div", id="Summary")
            if summary_block:
                indent_div = summary_block.find("div", class_="indent")
                if indent_div:
                    summary_text = indent_div.get_text(strip=True)

            return {
                "cwe_id": cwe_id,
                "title": title,
                "description": summary_text,
                "extended_description": "",
                "common_consequences": [],
            }

        # 获取 description
        desc = soup.find("div", attrs={"id": "Description"})
        if not desc:
            desc_tag = soup.find(string="Description")
            if desc_tag:
                desc = desc_tag.find_next("div")
        description_text = desc.get_text(separator="\n", strip=True) if desc else ""

        # 获取 extended description
        ext_desc = soup.find("div", attrs={"id": "Extended_Description"})
        if not ext_desc:
            ext_desc_tag = soup.find(string="Extended_Description")
            if ext_desc_tag:
                ext_desc = ext_desc_tag.find_next("div")
        EXdescription_text = ext_desc.get_text(separator="\n", strip=True) if ext_desc else ""

        # 获取 common_consequences
        results = []
        consequences_table = soup.find("table", id="Detail")
        if consequences_table:
            rows = consequences_table.find_all("tr")[1:]  # 跳过表头
            for row in rows:
                cols = row.find_all("td")
                if len(cols) == 2:
                    impact_text = cols[0].get_text(strip=True)
                    scope_span = cols[1].find("span", class_="suboptheading")
                    scope_text = scope_span.get_text(strip=True).replace("Scope: ", "") if scope_span else ""

                    results.append({
                        "Impact": impact_text,
                        "Scope": scope_text
                    })

        return {
            "cwe_id": cwe_id,
            "title": title,
            "description": description_text,
            "extended_description": EXdescription_text,
            "common_consequences": results
        }
    except Exception as e:
        print(f"[ERROR] 获取 CWE 数据失败 {cwe_id}: {e}")
        return None


# ================== 主程序 ===================
def main():
    df = pd.read_excel("datasets/train/train_all.xlsx")

    new_rows = []
    for idx, row in df.iterrows():
        cve_id = row["cve_id"]
        print(f"[INFO] 处理 {idx+1}/{len(df)}: {cve_id}")

        nvd_data = fetch_nvd_info(cve_id)
        if not nvd_data:
            continue

        cwe_info_list = []
        for cwe_id in nvd_data.get("cwe_ids", []):
            cwe_info = fetch_cwe_info(cwe_id)
            if cwe_info:
                cwe_info_list.append(cwe_info)
            time.sleep(1)

        # 更新 dataframe 行
        row_data = row.to_dict()
        row_data["cwe_ids"] = ",".join(nvd_data.get("cwe_ids", []))
        row_data["nvd_info"] = json.dumps({
            "cvss_v3x": nvd_data.get("cvss_v3x", {}),
            "affected_cpes": nvd_data.get("affected_cpes", [])
        }, ensure_ascii=False)
        row_data["cwe_info"] = json.dumps(cwe_info_list, ensure_ascii=False)

        new_rows.append(row_data)

        time.sleep(1)

    # 生成新 DataFrame
    new_df = pd.DataFrame(new_rows)

    # 调整列顺序：cve_id 后插入 cwe_ids，最后加 nvd_info、cwe_info
    cols = list(df.columns)
    insert_pos = cols.index("cve_id") + 1
    final_cols = cols[:insert_pos] + ["cwe_ids"] + cols[insert_pos:] + ["nvd_info", "cwe_info"]
    new_df = new_df[final_cols]

    new_df.to_excel("knowledge/train_all_with_nvd_cwe.xlsx", index=False)
    print("✅ 已写入 train_all_with_nvd_cwe.xlsx")


if __name__ == "__main__":
    main()
