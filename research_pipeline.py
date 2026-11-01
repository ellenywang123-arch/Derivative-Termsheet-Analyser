"""Team C research pipeline, sanitised from the supplied notebook.
Credentials and automatic installation/upload entry points have been removed.
Full LLM/PDF workflow is experimental and requires independent validation.
"""
import os
from IPython.display import display, HTML, Markdown, clear_output

class Config:
    API_KEY = os.environ.get("TERMSHEET_API_KEY", "")
    BASE_URL = os.environ.get("TERMSHEET_BASE_URL", "https://api.siliconflow.cn/v1")
    MODEL = os.environ.get("TERMSHEET_MODEL", "deepseek-ai/DeepSeek-V3.2")
    MAX_CHARS = 70000

import sys

import subprocess

import json

import traceback

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib.dates as mdates

from datetime import datetime, timedelta

import re

import os

import time

import textwrap

import random

import concurrent.futures

from itertools import cycle

from typing import List, Dict, Tuple, Optional

import fitz

from openai import OpenAI

import seaborn as sns

from graphviz import Digraph

class StreamDisplay:
    """
    实现类似 ChatGPT 的流式打字机输出效果，无闪烁。
    """
    @staticmethod
    def write(text, speed=0.01, is_markdown=True, color=None):
        """
        Stream text output without flickering.
        :param text: 要输出的文本
        :param speed: 每个字符的延迟时间 (秒)
        :param is_markdown: 是否解析 Markdown
        :param color: 文本颜色 (Hex code or name), e.g., "#00FF00"
        """
        # 尝试检测是否在 Jupyter/Colab 环境
        try:
            from IPython.display import display, Markdown, HTML
            in_notebook = True
        except ImportError:
            in_notebook = False
            print(text)
            return

        if not in_notebook:
            # 控制台回退模式
            for char in text:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(speed)
            print()
            return

        # 样式处理
        if color:
            text = f"<span style='color:{color}'>{text}</span>"
            is_markdown = False # 如果用了 span，最好转为 HTML 处理，或者外层包 Markdown

        # 创建一个唯一的显示句柄 (Handle)
        # 初始化显示一个光标
        cursor = " ▍"
        if is_markdown:
            empty_obj = Markdown(cursor)
        else:
            empty_obj = HTML(cursor)

        handle = display(empty_obj, display_id=True)

        buffer = ""
        # 稍微随机化速度，模拟真实打字感
        base_speed = speed

        for i, char in enumerate(text):
            buffer += char

            # 为了性能，不要每个字符都 update，每隔几个字符更新一次 DOM
            # 或者遇到换行符必须更新
            if i % 3 == 0 or char == '\n' or i == len(text) - 1:
                current_content = buffer + cursor
                if is_markdown:
                    handle.update(Markdown(current_content))
                else:
                    handle.update(HTML(current_content))

                # 随机延迟，看起来更像 AI 在思考
                time.sleep(max(0.002, base_speed + random.uniform(-0.005, 0.005)))

        # 最后移除光标
        if is_markdown:
            handle.update(Markdown(buffer))
        else:
            handle.update(HTML(buffer))

import time

class BlockTimer:
    """
    上下文管理器：用于测量代码块执行时间，并使用流式输出显示结果。
    """
    def __init__(self, name):
        self.name = name
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.time()
        # 开始时提示一下（可选，如果不想太啰嗦可以注释掉这一行）
        # StreamDisplay.write(f"⏳ Starting: {self.name}...", speed=0.01, color="#888")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type:
            # 如果报错了
            StreamDisplay.write(f"❌ {self.name} Failed after {elapsed:.2f}s", color="#ff4444")
        else:
            # 正常结束，显示绿色耗时
            StreamDisplay.write(f"⏱️ {self.name} Completed in {elapsed:.2f}s", speed=0.02, color="#4caf50")

def extract_structural_tags(text: str) -> Dict[str, bool]:
    text_lower = text.lower()
    tags = {
        "is_step_down": bool(re.search(r"step\s*[-]?\s*down|decreas(e|ing)\s*barrier", text_lower)),
        "is_memory": bool(re.search(r"memory|accumulat(e|ing)\s*coupon", text_lower)),
        "is_worst_of": bool(re.search(r"worst\s*[-]?\s*of|wo\b|lowest\s*performing", text_lower)),
        "is_phoenix": bool(re.search(r"phoenix", text_lower)),
        "is_autocall": bool(re.search(r"auto[-]?call|automatic\s*early\s*redemption", text_lower)),
        "frequency": "Unknown"
    }

    if re.search(r"monthly", text_lower):
        tags["frequency"] = "Monthly"
    elif re.search(r"quarterly", text_lower):
        tags["frequency"] = "Quarterly"
    elif re.search(r"semi-?annual", text_lower):
        tags["frequency"] = "Semi-Annual"
    elif re.search(r"daily", text_lower):
        tags["frequency"] = "Daily"

    return tags

def read_pdf_text(file_path_or_bytes, max_chars: int = Config.MAX_CHARS) -> str:
    try:
        doc = None
        if isinstance(file_path_or_bytes, str):
            doc = fitz.open(file_path_or_bytes)
        else:
            doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")

        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        text = "\n".join(parts)
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]

def fallback_extract_issuer(text: str) -> str:
    patterns = [
        r"(?im)^\s*Issued\s+by\s*:\s*(.+)$",
        r"(?im)^\s*Issued\s+by\s+(.+)$",
        r"(?im)^\s*Issuer\s*:\s*(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            issuer = m.group(1).strip()
            issuer = re.split(r"\s{2,}|\n", issuer)[0].strip()
            return issuer[:180]
    return "Unknown"

TASK1_SYSTEM_PROMPT = r"""
You are a Senior Derivatives Structurer.
TASK 1 ONLY: Extract quantified parameters from a term sheet.

ABSOLUTE OUTPUT RULES:
- Output ONLY TWO markdown tables. No extra text.
- KEY_TERMS_TABLE MUST include "Issuer" as the FIRST row.
- If a number/level/rate is shown ONLY as an example/illustration/hypothetical scenario, you MUST label it in Value as: "(Hypothetical)".

DATES_TABLE RULES:
- Include a specific column "Trigger / Barrier Level" to show the quantified percentage (e.g., "100%", "80%", "60%") associated with that observation date.
- The "Trigger / Barrier Level" column may contain multiple values if the date applies to multiple events.
- **CRITICAL**: In the "Applies To" column, you MUST strictly distinguish between "Autocall", "Coupon", and "Knock-in". DO NOT lump them together if possible.
- If multiple observation dates exist, list EACH date on its own row.
- Do not miss date types, for example, do not miss coupon payment dates if it exist.

FORMAT (EXACT):
KEY_TERMS_TABLE
| Term | Value |
| --- | --- |
| Issuer | ... |
...

DATES_TABLE
| Date Type | Date | Applies To | Trigger / Barrier Level | Reference Level / Notes |
| --- | --- | --- | --- | --- |
| Observation Date | ... | ... | ... | ... |
...
"""

TASK1_USER_PROMPT = r"""
Extract Task 1 parameters from the term sheet text below.

Return EXACTLY TWO TABLES:

KEY_TERMS_TABLE
| Term | Value |
| --- | --- |
| Issuer | ... |
| Currency | ... |
| Principal Amount | ... |
| Denomination | ... |
| Issue Price | ... |
| Underlyings | ... |
| Initial Level / Initial Price | ... |
| Strike Level / Strike Price | ... |
| Autocall Barrier / Call Value | ... (include step-down schedule if any) |
| Coupon Rate / Coupon Amount | ... |
| Coupon Barrier / Distribution Barrier | ... |
| Knock-in / Kick-in / Protection Barrier | ... |
| Cap / Floor / Buffer / Participation | ... |
(Include other essential quantified levels if present; but omit IDs and guarantor.)

DATES_TABLE
| Date Type | Date | Applies To | Trigger / Barrier Level | Reference Level / Notes |
| --- | --- | --- | --- | --- |
| Trade Date | ... | ... | ... | ... |
| Issue Date | ... | ... | ... | ... |
| Observation Date | ... | ... | ... | ... |
| Coupon Payment Date | ... | ... | ... | ... |
| Final Valuation Date | ... | ... | ... | ... |
| Maturity Date | ... | ... | ... | ... |
(CRITICAL INSTRUCTION: You MUST extract 'Coupon Payment Date' or 'Interest Payment Date'. If the dates are not explicitly listed but defined as a rule (e.g., "5 business days after Observation Date"), you MUST WRITE THE RULE in the "Date" column. DO NOT OMIT COUPON DATES. Populate "Trigger / Barrier Level" with the specific % required.)
TERM SHEET TEXT:
---
{TEXT}
---
"""

def call_task1_llm(client, text: str) -> str:
    resp = client.chat.completions.create(
        model=Config.MODEL,
        messages=[
            {"role": "system", "content": TASK1_SYSTEM_PROMPT},
            {"role": "user", "content": TASK1_USER_PROMPT.format(TEXT=text)},
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()

def extract_table_block(raw: str, table_name: str) -> str:
    pattern = rf"{table_name}\s*\n(\|.*\n)+"
    m = re.search(pattern, raw + "\n")
    return m.group(0) if m else ""

def markdown_table_to_df(table_block: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in table_block.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()
    if not lines[0].startswith("|"):
        lines = lines[1:]
    lines = [ln for ln in lines if ln.startswith("|")]
    if len(lines) < 2:
        return pd.DataFrame()

    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) < len(header):
            cols += [""] * (len(header) - len(cols))
        rows.append(cols[:len(header)])
    return pd.DataFrame(rows, columns=header)

def ensure_issuer_row(key_df: pd.DataFrame, issuer_fallback: str) -> pd.DataFrame:
    if key_df is None or key_df.empty or "Term" not in key_df.columns or "Value" not in key_df.columns:
        return pd.DataFrame([["Issuer", issuer_fallback]], columns=["Term", "Value"])

    key_df["Term"] = key_df["Term"].astype(str).str.strip()
    key_df["Value"] = key_df["Value"].astype(str).str.strip()

    mask = key_df["Term"].str.lower().eq("issuer")
    if not mask.any():
        key_df = pd.concat([pd.DataFrame([{"Term": "Issuer", "Value": issuer_fallback}]), key_df], ignore_index=True)
    else:
        idx = key_df[mask].index[0]
        if (not key_df.loc[idx, "Value"]) or key_df.loc[idx, "Value"].lower() == "unknown":
            key_df.loc[idx, "Value"] = issuer_fallback
        issuer_row = key_df.loc[[idx]]
        rest = key_df.drop(index=idx)
        key_df = pd.concat([issuer_row, rest], ignore_index=True)
    return key_df

SYNONYM_GROUPS = {
    "Knock-in Barrier": [
        "protection barrier", "knock-in", "kick-in", "ki barrier", "downside barrier",
        "safety barrier", "protection level", "risk barrier"
    ],
    "Coupon Barrier": [
        "coupon barrier", "distribution barrier", "interest barrier", "coupon threshold",
        "yield barrier", "conditional coupon level"
    ],
    "Autocall Level": [
        "autocall barrier", "call value", "autocall condition", "automatic early redemption threshold",
        "early redemption level"
    ],
    "Initial/Strike": [
        "initial level", "initial price", "strike level", "strike price", "reference price"
    ],
}

def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_value(v: str) -> str:
    v = (v or "").strip().lower()
    v = v.replace(" ", "")
    v = v.replace(",", "")
    return v

def merge_synonyms_same_value(key_df: pd.DataFrame) -> pd.DataFrame:
    if key_df is None or key_df.empty:
        return key_df
    if "Term" not in key_df.columns or "Value" not in key_df.columns:
        return key_df

    issuer_row = key_df[key_df["Term"].astype(str).str.lower().eq("issuer")]
    rest = key_df[~key_df["Term"].astype(str).str.lower().eq("issuer")].copy()

    def canonical(term: str) -> str:
        t = _norm_text(term)
        for canon, kws in SYNONYM_GROUPS.items():
            if any(kw in t for kw in kws):
                return canon
        return term.strip()

    rest["__canon__"] = rest["Term"].astype(str).apply(canonical)
    rest["__vnorm__"] = rest["Value"].astype(str).apply(_norm_value)

    merged_rows = []
    used = set()

    for i, row in rest.iterrows():
        if i in used:
            continue
        canon = row["__canon__"]
        vnorm = row["__vnorm__"]
        same = rest[(rest["__canon__"] == canon) & (rest["__vnorm__"] == vnorm)]

        idxs = list(same.index)
        for j in idxs:
            used.add(j)

        if len(idxs) == 1 or canon == row["Term"].strip():
            term_label = row["Term"].strip()
        else:
            aliases = [rest.loc[j, "Term"].strip() for j in idxs]
            seen = set()
            aliases2 = []
            for a in aliases:
                if a.lower() not in seen:
                    seen.add(a.lower())
                    aliases2.append(a)
            term_label = f"{canon} ({' / '.join(aliases2)})"

        merged_rows.append({"Term": term_label, "Value": row["Value"].strip()})

    out = pd.DataFrame(merged_rows)
    if issuer_row is not None and not issuer_row.empty:
        out = pd.concat([issuer_row[["Term", "Value"]], out], ignore_index=True)

    return out[["Term", "Value"]]

_FREQ_WORDS = re.compile(r"\b(quarterly|monthly|weekly|annually|semi-?annually|bi-?weekly)\b", re.I)

def enforce_observation_dates(dates_df: pd.DataFrame, original_text: str) -> pd.DataFrame:
    if dates_df is None or dates_df.empty:
        return dates_df

    if "Date" not in dates_df.columns or "Date Type" not in dates_df.columns:
        return dates_df

    date_patterns = [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]
    all_dates = []
    for pat in date_patterns:
        all_dates += re.findall(pat, original_text, flags=re.I)
    seen = set()
    explicit_dates = []
    for d in all_dates:
        dd = d.strip()
        if dd not in seen:
            seen.add(dd)
            explicit_dates.append(dd)

    obs_mask = dates_df["Date Type"].astype(str).str.lower().str.contains("observation")
    if not obs_mask.any():
        return dates_df

    for idx in list(dates_df[obs_mask].index):
        date_val = str(dates_df.loc[idx, "Date"]).strip()
        if _FREQ_WORDS.search(date_val):
            applies = dates_df.loc[idx, "Applies To"]
            notes = dates_df.loc[idx, "Reference Level / Notes"] if "Reference Level / Notes" in dates_df.columns else ""

            trigger = "See Key Terms"
            col_candidates = ["Trigger / Barrier Level", "Trigger Levels", "Trigger Level", "Autocall / Barrier Level"]
            for col in col_candidates:
                if col in dates_df.columns:
                    trigger = dates_df.loc[idx, col]
                    break

            dates_df = dates_df.drop(index=idx)

            if explicit_dates:
                new_rows = []
                for d in explicit_dates:
                    row = {
                        "Date Type": "Observation Date",
                        "Date": d,
                        "Applies To": applies,
                        "Trigger / Barrier Level": trigger,
                        "Reference Level / Notes": notes if notes else "Explicit date extracted"
                    }
                    new_rows.append(row)
                dates_df = pd.concat([dates_df, pd.DataFrame(new_rows)], ignore_index=True)
            else:
                row = {
                    "Date Type": "Observation Date",
                    "Date": "Not explicitly specified",
                    "Applies To": applies,
                    "Trigger / Barrier Level": trigger,
                    "Reference Level / Notes": f"Frequency stated: {date_val}"
                }
                dates_df = pd.concat([dates_df, pd.DataFrame([row])], ignore_index=True)

    return dates_df

def compress_dates_table(dates_df: pd.DataFrame) -> pd.DataFrame:
    if dates_df is None or dates_df.empty:
        return dates_df

    if "Date Type" not in dates_df.columns or "Date" not in dates_df.columns:
        return dates_df

    date_types_order = []
    for v in dates_df["Date Type"].astype(str):
        if v not in date_types_order:
            date_types_order.append(v)

    grouped_rows = []

    for dt in date_types_order:
        sub = dates_df[dates_df["Date Type"].astype(str) == str(dt)].copy()

        date_lines = sub["Date"].astype(str).tolist()
        dates_str = "<br>".join(date_lines)

        if "Trigger / Barrier Level" in sub.columns:
            trig_lines = sub["Trigger / Barrier Level"].astype(str).tolist()
            clean_trigs = [
                t for t in trig_lines
                if t and t.strip() and t.lower() not in ("nan", "n/a", "none")
            ]

            if len(clean_trigs) == 0:
                trig_str = ""
            elif len(set(clean_trigs)) == 1:
                trig_str = clean_trigs[0]
            else:
                trig_str = "<br>".join(trig_lines)
        else:
            trig_str = ""

        if "Reference Level / Notes" in sub.columns:
            notes_lines = sub["Reference Level / Notes"].astype(str).tolist()
            clean_notes = [
                n for n in notes_lines
                if n and n.strip() and n.lower() not in ("nan", "n/a", "none")
            ]

            if len(clean_notes) == 0:
                notes_str = ""
            elif len(set(clean_notes)) == 1:
                notes_str = clean_notes[0]
            else:
                notes_str = "<br>".join(notes_lines)
        else:
            notes_str = ""

        if "Applies To" in sub.columns:
            applies_vals = [
                x for x in sub["Applies To"].astype(str).tolist()
                if x and x.strip() and x.lower() not in ("nan", "n/a", "none")
            ]
            seen = set()
            applies_list = []
            for a in applies_vals:
                if a not in seen:
                    seen.add(a)
                    applies_list.append(a)
            applies_str = ", ".join(applies_list)
        else:
            applies_str = ""

        grouped_rows.append({
            "Date Type": dt,
            "Dates": dates_str,
            "Applies To": applies_str,
            "Trigger / Barrier Level": trig_str,
            "Reference Level / Notes": notes_str,
        })

    out_cols = ["Date Type", "Dates", "Applies To", "Trigger / Barrier Level", "Reference Level / Notes"]
    return pd.DataFrame(grouped_rows, columns=out_cols)

def style_task1_table(df: pd.DataFrame, caption: str):
    def zebra(row):
        bg = "#f9f9f9" if row.name % 2 == 0 else "#ffffff"
        return [f"background-color: {bg}"] * len(row)

    styler = (
        df.style
          .hide(axis="index")
          .set_caption(caption)
          .set_table_styles(
              [
                  {
                      "selector": "caption",
                      "props": [
                          ("caption-side", "top"),
                          ("text-align", "left"),
                          ("font-weight", "bold"),
                          ("font-size", "14px"),
                          ("background-color", "#007ACC"),
                          ("color", "white"),
                          ("padding", "8px"),
                          ("border-radius", "4px 4px 0 0"),
                      ],
                  },
                  {
                      "selector": "table",
                      "props": [
                          ("border-collapse", "collapse"),
                          ("border", "1px solid #d0d0d0"),
                          ("width", "100%"),
                          ("font-size", "12px"),
                          ("background-color", "white"),
                      ],
                  },
                  {
                      "selector": "th",
                      "props": [
                          ("text-align", "left"),
                          ("padding", "6px 10px"),
                          ("background-color", "#f2f2f2"),
                          ("border-bottom", "1px solid #d0d0d0"),
                          ("white-space", "pre-wrap"),
                      ],
                  },
                  {
                      "selector": "td",
                      "props": [
                          ("text-align", "left"),
                          ("padding", "6px 10px"),
                          ("border-bottom", "1px solid #eeeeee"),
                          ("white-space", "pre-wrap"),
                      ],
                  },
              ],
              overwrite=False,
          )
          .apply(zebra, axis=1)
          .set_properties(**{"text-align": "left", "padding": "6px 10px"})
    )
    return styler

def filter_key_terms_core(key_df: pd.DataFrame) -> pd.DataFrame:
    """
    精简版 Key Terms：
    - 保留与 Payoff / 交易直接相关的核心参数
    - 过滤掉 listing / code / marketing / admin 等无关字段
    """
    if key_df is None or key_df.empty:
        return key_df

    keep_rows = []
    core_keywords = [
        "issuer",
        "underlying", "underlyings",
        "currency",
        "principal", "notional", "nominal",
        "denomination",
        "issue price", "initial price",
        "initial level", "initial fixing", "initial reference",
        "strike",
        "autocall", "early redemption", "call value",
        "coupon", "distribution", "interest",
        "barrier", "knock-in", "kick-in", "knock-out", "protection",
        "buffer",
        "cap", "floor", "participation",
        "tenor", "maturity", "final redemption"
    ]

    blacklist_keywords = [
        "listing", "stock exchange", "ric", "isin", "code",
        "common code", "wkn", "sedol",
        "subscription", "offer period",
        "selling restriction", "target market",
        "marketing", "disclaimer", "documentation"
    ]

    for _, row in key_df.iterrows():
        term = str(row.get("Term", "")).strip()
        norm = term.lower()

        if norm == "issuer":
            keep_rows.append(True)
            continue

        if any(bad in norm for bad in blacklist_keywords):
            keep_rows.append(False)
            continue

        if any(kw in norm for kw in core_keywords):
            keep_rows.append(True)
        else:
            keep_rows.append(False)

    filtered = key_df.loc[keep_rows].reset_index(drop=True)
    return filtered

def filter_dates_for_summary(dates_df: pd.DataFrame) -> pd.DataFrame:
    """
    精简版 Dates：
    - 只保留与投资 / Payoff 相关的日期类型
      (Trade, Issue, Fixing / Valuation, Observation, Early Redemption/Autocall,
       Coupon Valuation / Coupon Payment, Interest Payment, Final Valuation, Maturity 等)
    - 过滤掉 Listing / Subscription / Marketing / Admin 类型日期
    - 保留 compress_dates_table 的格式：同一 Date Type 一行，Dates 用 <br> 换行
    """
    if dates_df is None or dates_df.empty:
        return dates_df
    if "Date Type" not in dates_df.columns:
        return dates_df

    keep_rows = []

    good_keywords = [
        "trade",          # Trade Date
        "issue",          # Issue Date
        "fixing", "strike",
        "valuation",      # Valuation Date / Final Valuation Date
        "observation",    # Observation / Autocall Observation
        "autocall", "auto-call",
        "early redemption", "redemption",
        "coupon",         # Coupon Valuation / Coupon Payment
        "interest",       # Interest Payment Date
        "payment",        # Payment Date
        "knock-in", "knock out", "knock-out",
        "maturity", "expiry", "expiration",
    ]

    bad_keywords = [
        "listing",        # Listing Date
        "stock exchange",
        "subscription",   # Subscription Period / Subscription Date
        "offer period",
        "marketing",
        "target market",
        "announcement",
        "publication",
        "delivery of documentation",
        "documentation",
        "record date",
    ]

    for _, row in dates_df.iterrows():
        dt_type = str(row.get("Date Type", "")).strip()
        norm = dt_type.lower()

        if any(bad in norm for bad in bad_keywords):
            keep_rows.append(False)
            continue

        if any(good in norm for good in good_keywords):
            keep_rows.append(True)
        else:
            keep_rows.append(False)

    filtered = dates_df.loc[keep_rows].reset_index(drop=True)
    return filtered

def render_task1_table(df: pd.DataFrame, caption: str):
    if df is None or df.empty:
        display(Markdown(f"_No data for **{caption}**._"))
        return

    try:
        styled = style_task1_table(df, caption)
        html = styled.to_html(escape=False)
        display(HTML(html))
    except Exception:
        display(Markdown(f"**{caption}**"))
        display(df)

def generate_constrained_path(start_val, end_val, n_points, vol=2.0, force_min=None, force_max=None, seed=None):
    """
    This function is now DEPRECATED for scenario visualization.
    Kept for backward compatibility.
    """
    if seed is not None:
        np.random.seed(seed)

    if start_val is None: start_val = 100.0
    if end_val is None: end_val = 100.0

    path = np.linspace(start_val, end_val, n_points)
    noise = np.random.normal(0, vol * 0.3, n_points)
    path = path + noise
    path[0] = start_val
    path[-1] = end_val

    return path

def generate_discrete_scenario_points(scenario_data, calendar, barrier_hierarchy, start_date, end_date):
    """
    Generate discrete price points at each observation date based on scenario logic.

    IMPORTANT:
    - Returns points for ALL observation dates (for x-axis display)
    - 'is_termination' marks where the product terminates
    - Points AFTER termination have 'price' = None (will not be plotted but date shown on x-axis)

    Returns: (points_list, total_obs_count, termination_obs_index)
    """
    points = []

    # Parse scenario information
    condition = str(scenario_data.get('condition', '')).lower()
    outcome = str(scenario_data.get('outcome', '')).lower()
    chart_type = str(scenario_data.get('chart_type', '')).lower()

    # Extract barrier levels
    ko_level = None
    ki_level = None
    coupon_level = None

    for b in barrier_hierarchy:
        b_type = str(b.get('type', '')).lower()
        try:
            val = float(re.sub(r'[^\d.\-]', '', str(b.get('value', ''))))
            if val < 5:
                val = val * 100
        except:
            continue

        if 'knock-out' in b_type or 'autocall' in b_type:
            ko_level = val
        elif 'knock-in' in b_type or 'protection' in b_type:
            ki_level = val
        elif 'coupon' in b_type:
            coupon_level = val

    # Default barriers if not found
    if ko_level is None:
        ko_level = 100.0
    if ki_level is None:
        ki_level = 60.0
    if coupon_level is None:
        coupon_level = ki_level if ki_level else 70.0

    # Collect ALL observation dates from calendar
    all_obs_dates = []
    for entry in calendar:
        try:
            d = datetime.strptime(entry.get('date', ''), '%Y-%m-%d')
            if start_date <= d <= end_date:
                levels = entry.get('levels', {})
                date_ko = ko_level
                for k, v in levels.items():
                    if 'knock-out' in k.lower() or 'autocall' in k.lower():
                        try:
                            date_ko = float(re.sub(r'[^\d.\-]', '', str(v)))
                            if date_ko < 5:
                                date_ko = date_ko * 100
                        except:
                            pass
                all_obs_dates.append({'date': d, 'ko_level': date_ko})
        except:
            continue

    # Sort ALL observations chronologically and assign sequential index
    all_obs_dates.sort(key=lambda x: x['date'])

    # Assign actual observation index (1-based)
    for idx, obs in enumerate(all_obs_dates):
        obs['obs_index'] = idx + 1

    total_obs_count = len(all_obs_dates)

    # If no observation dates found, create some
    if not all_obs_dates:
        total_days = (end_date - start_date).days
        for i in range(1, 32):  # 31 observations
            d = start_date + timedelta(days=int(total_days * i / 31))
            if d <= end_date:
                all_obs_dates.append({'date': d, 'ko_level': ko_level, 'obs_index': i})
        total_obs_count = len(all_obs_dates)

    # Add start point (not an observation, obs_index = 0)
    points.append({
        'date': start_date,
        'price': 100.0,
        'is_termination': False,
        'termination_reason': None,
        'obs_index': 0,
        'is_after_termination': False
    })

    # ============================================================
    # SCENARIO TYPE DETECTION - IMPROVED LOGIC
    # Priority order matters: check more specific conditions first
    # ============================================================

    # 1. Check for Autocall (Early Redemption)
    is_autocall = (
        'autocall' in chart_type or
        'autocall' in condition or
        ('early' in condition and 'redemption' in condition) or
        ('early' in condition and 'call' in condition)
    )

    # 2. Check for KI Recovery (V-shape rebound) - must check BEFORE ki_loss
    is_ki_recovery = (
        'ki_recovery' in chart_type or
        'recovery' in chart_type or
        'rebound' in condition or
        'recover' in condition or
        'v-shape' in condition or
        ('knock-in' in condition and 'recover' in condition)
    )

    # 3. Check for KI Loss - only if NOT recovery
    is_ki_loss = (
        not is_ki_recovery and
        (
            'ki_loss' in chart_type or
            'loss' in chart_type or
            ('knock-in' in condition and 'loss' in condition) or
            ('breach' in condition and 'loss' in outcome) or
            ('touched' in condition and 'loss' in outcome)
        )
    )

    # 4. Check for Safe Maturity - explicitly check for safe/par/no knock-in
    is_safe_maturity = (
        not is_autocall and
        not is_ki_loss and
        not is_ki_recovery and
        (
            'safe' in chart_type or
            'safe' in condition or
            'at par' in condition or
            'no knock-in' in condition or
            'above' in condition or
            ('maturity' in condition and 'loss' not in condition and 'knock-in' not in condition) or
            'full principal' in outcome or
            'principal protected' in outcome or
            '100%' in outcome
        )
    )

    # 5. Default: if none of above, check outcome for hints
    if not is_autocall and not is_ki_loss and not is_ki_recovery and not is_safe_maturity:
        # Check outcome for final determination
        if 'loss' in outcome or 'below' in outcome:
            is_ki_loss = True
        else:
            # Default to safe maturity
            is_safe_maturity = True

    np.random.seed(hash(condition) % 2**32)

    termination_obs_index = None

    # Determine termination point based on scenario type
    if is_autocall:
        # Find which observation triggers autocall
        month_match = re.search(r'month\s*(\d+)', condition)
        year_match = re.search(r'year\s*(\d+)', condition)

        if month_match:
            target_month = int(month_match.group(1))
            termination_obs_index = min(target_month, total_obs_count)
        elif year_match:
            target_year = int(year_match.group(1))
            termination_obs_index = min(target_year * 12, total_obs_count)
        else:
            termination_obs_index = max(1, total_obs_count // 3)
    else:
        # All other scenarios terminate at maturity (last observation)
        termination_obs_index = total_obs_count

    # ============================================================
    # GENERATE PRICES FOR ALL OBSERVATION DATES
    # ============================================================
    for i, obs in enumerate(all_obs_dates):
        actual_obs_idx = obs['obs_index']  # 1-based
        date_ko = obs['ko_level']
        is_after_termination = actual_obs_idx > termination_obs_index
        is_termination_point = actual_obs_idx == termination_obs_index

        # Calculate price based on scenario type
        if is_after_termination:
            price = None
            term_reason = None

        elif is_autocall:
            if actual_obs_idx < termination_obs_index:
                base_price = date_ko - np.random.uniform(5, 15)
                noise = np.random.uniform(-3, 3)
                price = max(ki_level + 5, base_price + noise)
                term_reason = None
            else:
                price = date_ko + np.random.uniform(2, 8)
                term_reason = f'Autocall Triggered (Obs {actual_obs_idx})'

        elif is_ki_loss:
            # Knock-in loss scenario
            drop_idx = total_obs_count // 3

            if actual_obs_idx < drop_idx:
                price = 100 - (actual_obs_idx / max(drop_idx, 1)) * (100 - ki_level - 5) + np.random.uniform(-3, 3)
                price = max(ki_level + 2, price)
            elif actual_obs_idx == drop_idx:
                price = ki_level - np.random.uniform(3, 10)
            else:
                progress = (actual_obs_idx - drop_idx) / max(total_obs_count - drop_idx, 1)
                end_target = scenario_data.get('end_price_target', ki_level - 20)
                try:
                    end_target = float(re.sub(r'[^\d.\-]', '', str(end_target)))
                    if end_target < 5:
                        end_target = end_target * 100
                except:
                    end_target = ki_level - 20

                price = ki_level - 5 + progress * (end_target - (ki_level - 5))
                price += np.random.uniform(-3, 3)

            term_reason = f'Maturity (Knock-in Loss, Obs {actual_obs_idx})' if is_termination_point else None

        elif is_ki_recovery:
            # V-shape recovery
            drop_idx = total_obs_count // 3
            recovery_start = total_obs_count // 2

            if actual_obs_idx < drop_idx:
                price = 100 - (actual_obs_idx / max(drop_idx, 1)) * (100 - ki_level + 5) + np.random.uniform(-2, 2)
            elif actual_obs_idx < recovery_start:
                price = ki_level - np.random.uniform(5, 15)
            else:
                progress = (actual_obs_idx - recovery_start) / max(total_obs_count - recovery_start, 1)
                price = (ki_level - 10) + progress * (105 - (ki_level - 10))
                price += np.random.uniform(-3, 3)

            if is_termination_point:
                price = max(price, 100 + np.random.uniform(2, 8))

            term_reason = f'Maturity (KI Recovery, Obs {actual_obs_idx})' if is_termination_point else None

        elif is_safe_maturity:
            # Safe maturity scenario: price stays above KI, ends at strike (100%)
            mid_point = (ki_level + date_ko) / 2
            amplitude = (date_ko - ki_level) / 4

            progress = actual_obs_idx / total_obs_count
            trend_target = 100.0

            base_price = mid_point * (1 - progress) + trend_target * progress
            noise = np.random.uniform(-amplitude * (1 - progress * 0.5), amplitude * (1 - progress * 0.5))
            price = base_price + noise

            price = max(ki_level + 3, min(date_ko - 3, price))

            # At maturity, redemption at strike (100%) - Safe/At Par
            if is_termination_point:
                price = 100.0

            # FIXED: Correct termination reason for safe maturity
            term_reason = f'Maturity (Safe/At Par, Obs {actual_obs_idx})' if is_termination_point else None

        else:
            # Fallback: treat as safe scenario
            price = 100 + np.random.uniform(-10, 10)
            price = max(ki_level + 3, min(date_ko - 3, price))
            if is_termination_point:
                price = 100.0
            term_reason = f'Maturity (At Par, Obs {actual_obs_idx})' if is_termination_point else None

        points.append({
            'date': obs['date'],
            'price': price,
            'is_termination': is_termination_point,
            'termination_reason': term_reason,
            'obs_index': actual_obs_idx,
            'is_after_termination': is_after_termination
        })

    return points, total_obs_count, termination_obs_index

class LogicArchitect:
    """Task 2 Logic: Analyzes structure using Task 1 Inputs + Text."""
    def __init__(self, config):
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.model = config.MODEL

    def analyze_structure(self, text, key_df=None, dates_df=None, extra_tags=None):
        terms_context = key_df.to_string() if key_df is not None and not key_df.empty else "No Key Terms extracted."
        dates_context = dates_df.to_string() if dates_df is not None and not dates_df.empty else "No Dates extracted."
        tags_context = json.dumps(extra_tags) if extra_tags else "No tags."

        prompt = r"""
        You are a Senior Derivatives Structurer.
        Perform a comprehensive structural analysis based on the extracted data.

        INPUT DATA (FROM PREVIOUS EXTRACTION):
        1. Extracted Key Terms:
        {TERMS}
        2. Extracted Dates & Triggers:
        {DATES}
        3. Structural Tags:
        {TAGS}

        MANDATORY INSTRUCTIONS:
        1. BARRIER HIERARCHY: Extract ALL barriers from 'Extracted Key Terms' (Knock-out, Knock-in, Coupon, Strike).
        2. DYNAMIC SCHEDULE: Construct the 'calendar' using the 'Extracted Dates' table.
           - Map 'Trigger / Barrier Level' from the dates table to the 'levels' object.
           - **CRITICAL ROW-CONTEXT DISAMBIGUATION**: The 'Trigger / Barrier Level' value MUST be interpreted based on the 'Applies To' column IN THE SAME ROW.
             * IF 'Applies To' contains "Autocall" AND "Knock-in" (e.g., "Autocall / Coupon"): Split the 'Trigger' value (e.g., "110% / 70%") and map 110%->Knock-out, 70%->Coupon.
             * IF 'Applies To' contains ONLY "Autocall" or "Early Redemption": The value is a "Knock-out" barrier.
             * IF 'Applies To' contains ONLY "Kick-in", "Protection", or "Expiration": The value is a "Knock-in" barrier.
           - **EXACT BARRIER SCHEDULE (ALL TYPES)**: You must populate the 'levels' for **EVERY** observation date individually.
             * **Variable/Step-Down**: If levels change (e.g., 100% -> 90%), you **MUST** record the specific level for that specific date. Do not lazy-load the initial value.
             * **Constant**: If levels are constant (e.g., always 100%), ensure this constant value is explicitly applied to each date entry.
        3. FORMULA EXTRACTION (CRITICAL):
           - Return mathematical formulas in LaTeX format.
           - In 'narrative', DO NOT just read out the math operations (e.g. "N times 100%").
           - Instead, explain the LOGIC and MECHANISM in plain English.
           - AIM FOR: "Your return tracks the weighted average performance of the basket assets, meaning each asset contributes proportional to its assigned weight."
           - AVOID: "The return is calculated by multiplying the principal by the final performance."
        4. **SCENARIO GENERATION (ADAPTIVE MECE - Mutually Exclusive, Collectively Exhaustive)**:
           - **DO NOT** force a fixed number of scenarios. The number of scenarios depends ENTIRELY on the **Product Structure**.
           - **RULE 1: DETERMINE SCENARIO COUNT BY STRUCTURE**:
             * **Capital Protected / Shark Fin**: Usually **2 Scenarios** (Base Case/Floor, Bull Case/Upside).
             * **Vanilla Autocall / Phoenix**: Usually **3 Scenarios** (Early Redemption, Safe Maturity, Loss Maturity).
             * **American Knock-In (Daily Observation)**: Requires **4-5 Scenarios**. You **MUST** include a **"V-Shape Rebound"** scenario (Touched KI barrier during term, but recovered > Strike at maturity).
           - **RULE 2: STRICT TIME-VALUE & COUPON LOGIC (CRITICAL)**:
             * A scenario ending in loss (Maturity Loss) usually means the product survived for the full term.
             * **YOU MUST CALCULATE**: Did the investor receive periodic coupons BEFORE the crash?
             * **Outcome Description**: MUST be explicit. E.g., "**Principal Loss + 8 Coupons (Received during the first 2 years)**".
             * NEVER just say "Loss" if coupons were paid periodically.
        5. **CORE STRUCTURE & RULES SUMMARY**:
           - **"core_structure"**: A single concise sentence summarizing the financial structure (e.g. "This is a 3-year autocallable note paying fixed monthly coupons of 0.5%...").
           - **"rules_logic"**: Explain the detailed mechanics in a clear, numbered format (e.g. "1) Fixed coupon payments are paid monthly... 2) Early redemption is triggered if... 3) Final redemption depends on...").
        6. **FLOWCHART DESIGN (Graphviz Structure)**:
           - You MUST generate a 'flowchart_design' object with 'nodes' and 'edges' to visualize the product lifecycle.
           - **Nodes**: List of objects {{"id": "str", "label": "str", "type": "start|action|decision|exit"}}.
             * 'start': Issue Date / Start.
             * 'action': Observation dates, Coupon Accumulation, etc.
             * 'decision': "Above Barrier?", "Knock-out Event?", "Knock-in Event?".
             * 'exit': Maturity, Early Redemption.
           - **Edges**: List of objects {{"from": "id", "to": "id", "label": "Yes/No/Next", "style": "solid|dashed"}}.
             * Use "dashed" style for loops (e.g. returning to waiting state or "No" path that leads to next observation).

        Output strictly in JSON format:
        {{
            "product_classification": {{
                "type": "Product Type Name",
                "confidence": 0.9,
                "reasoning": "Brief reasoning"
            }},
            "product_id": "Product Name",
            "core_structure": "This is a 3-year structured note...",
            "rules_logic": "1) Coupon mechanics... 2) Autocall triggers... 3) Final redemption...",
            "terms_purpose": "Purpose...",
            "dates": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
            "barrier_hierarchy": [
                {{"name": "Early Redemption Level", "type": "Knock-out", "value": 100.0}},
                {{"name": "Coupon Barrier", "type": "Coupon", "value": 70.0}}
            ],
            "calendar": [
                {{"date": "YYYY-MM-DD", "event": "Observation", "levels": {{"Knock-out": 95.0, "Coupon": 70.0}}}}
            ],
            "payoff_logic": {{
                "natural_language": "Logic summary...",
                "formulas": [
                    {{
                        "label": "Final Redemption (Scenario A)",
                        "latex": "N \\times 100\\%".
                        "narrative": "Investors receive full principal repayment because..."
                    }}
                ],
                "mece_scenarios": [
                    {{
                        "condition": "Early Autocall (Month 7)",
                        "outcome": "Principal + 7 Coupons",
                        "chart_type": "autocall",
                        "termination_type": "autocall",
                        "termination_month": 7,
                        "end_price_target": 105.0
                    }},
                    {{
                        "condition": "Safe Maturity (No Knock-in)",
                        "outcome": "Principal + 36 Coupons",
                        "chart_type": "safe_maturity",
                        "termination_type": "maturity_safe",
                        "termination_month": null,
                        "end_price_target": 85.0
                    }},
                    {{
                        "condition": "Knock-in Loss",
                        "outcome": "Principal Loss + Coupons Received",
                        "chart_type": "ki_loss",
                        "termination_type": "maturity_loss",
                        "termination_month": null,
                        "end_price_target": 45.0
                    }}
                ]
            }},
            "flowchart_design": {{
                "nodes": [
                    {{"id": "n1", "label": "Start", "type": "start"}},
                    {{"id": "n2", "label": "Observation", "type": "decision"}}
                ],
                "edges": [
                    {{"from": "n1", "to": "n2", "label": "Next"}}
                ]
            }}
        }}
        """.format(TERMS=terms_context, DATES=dates_context, TAGS=tags_context)

        max_retries = 3
        viz_data = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                    temperature=0.1
                )
                content = response.choices[0].message.content
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_match:
                    viz_data = json.loads(json_match.group(1).strip())
                    break
                elif "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                    viz_data = json.loads(content.strip())
                    break
            except Exception:
                time.sleep(1)

        return viz_data

class LifecycleFlowVisualizer:
    """
    通用生命周期逻辑流可视化 (基于 Graphviz 动态渲染)
    """

    def __init__(self):
        self.styles = {
            'graph': {
                'rankdir': 'TB',
                'splines': 'ortho',
                'nodesep': '0.6',
                'ranksep': '0.6',
                'fontname': 'Helvetica',
            },
            'types': {
                'start': {
                    'shape': 'box', 'style': 'filled,rounded',
                    'fillcolor': '#e1f5fe', 'color': '#01579b',
                    'fontname': 'Helvetica-Bold', 'fontsize': '10'
                },
                'action': {
                    'shape': 'note',
                    'style': 'filled',
                    'fillcolor': '#fff9c4', 'color': '#fbc02d',
                    'fontname': 'Helvetica', 'fontsize': '10'
                },
                'decision': {
                    'shape': 'diamond', 'style': 'filled',
                    'fillcolor': '#fff3e0', 'color': '#e65100',
                    'fontname': 'Helvetica', 'fontsize': '9'
                },
                'exit': {
                    'shape': 'doubleoctagon', 'style': 'filled',
                    'fillcolor': '#e8f5e9', 'color': '#2e7d32',
                    'fontname': 'Helvetica-Bold', 'fontsize': '10'
                },
                'default': {
                    'shape': 'box', 'style': 'filled',
                    'fillcolor': '#f5f5f5', 'color': '#9e9e9e'
                }
            },
            'edge': {
                'fontname': 'Helvetica',
                'fontsize': '8',
                'color': '#424242',
                'arrowsize': '0.7'
            }
        }

    def plot_logic_flow(self, viz_data, save_path=None):
        dot = Digraph(comment='Dynamic Lifecycle Flow')
        dot.attr(**self.styles['graph'])
        dot.edge_attr.update(self.styles['edge'])

        flow_design = viz_data.get('flowchart_design', {})
        if not flow_design:
            return None

        nodes = flow_design.get('nodes', [])
        edges = flow_design.get('edges', [])
        existing_node_ids = set()

        for node in nodes:
            n_id = node.get('id')
            n_label = node.get('label', 'Node')
            n_type = node.get('type', 'default').lower()

            if not n_id: continue

            style_attrs = self.styles['types'].get(n_type, self.styles['types']['default']).copy()
            clean_label = str(n_label).replace('\\n', '\n')

            dot.node(n_id, label=clean_label, **style_attrs)
            existing_node_ids.add(n_id)

        for edge in edges:
            src = edge.get('from')
            dst = edge.get('to')
            label = edge.get('label', '')
            style = edge.get('style', 'solid')

            if src in existing_node_ids and dst in existing_node_ids:
                attrs = {}
                if label: attrs['xlabel'] = label
                if style == 'dashed':
                    attrs['style'] = 'dashed'
                    attrs['constraint'] = 'false'

                dot.edge(src, dst, **attrs)

        if save_path is None:
            rnd_id = random.randint(10000, 99999)
            save_path = f"dynamic_flow_{rnd_id}"

        base_path = os.path.splitext(save_path)[0]

        try:
            output_path = dot.render(filename=base_path, format='png', cleanup=True, view=False)
            return output_path
        except Exception as e:
            print(f"Graphviz rendering failed: {e}")
            return None

class EducationalVisualizer:
    """Visualizes scenarios with discrete points at observation dates."""

    BARRIER_CONFIG = {
        'Knock-out': {'color': '#228B22', 'style': '--', 'label': 'Knock-out / Autocall', 'zorder': 4, 'lw': 2.0},
        'Knock-in': {'color': '#DC143C', 'style': '-', 'label': 'Knock-in / Protection', 'zorder': 2, 'lw': 2.0},
        'Coupon': {'color': '#4169E1', 'style': '-.', 'label': 'Coupon Barrier', 'zorder': 3, 'lw': 1.5},
        'Strike': {'color': '#696969', 'style': ':', 'label': 'Strike (100%)', 'zorder': 5, 'lw': 1.2}
    }

    # Standardized Y-axis range
    Y_AXIS_MIN = 0
    Y_AXIS_MAX = 110

    def _normalize_value(self, val: float) -> float:
        if val is None:
            return None
        try:
            val = float(val)
        except:
            return None
        if np.isnan(val):
            return None
        if val == 0:
            return 0.0
        if abs(val) < 5.0:
            return val * 100.0
        return val

    def _extract_barrier_levels(self, barrier_hierarchy):
        """Extract and normalize barrier levels from hierarchy."""
        levels = {'Knock-out': None, 'Knock-in': None, 'Coupon': None}

        for b in barrier_hierarchy:
            b_type = str(b.get('type', '')).lower()
            try:
                val = float(re.sub(r'[^\d.\-]', '', str(b.get('value', ''))))
                val = self._normalize_value(val)
            except:
                continue

            if 'knock-out' in b_type or 'autocall' in b_type:
                levels['Knock-out'] = val
            elif 'knock-in' in b_type or 'protection' in b_type:
                levels['Knock-in'] = val
            elif 'coupon' in b_type:
                levels['Coupon'] = val

        return levels

    def _draw_step_barrier(self, ax, x_min, x_max, schedule, initial_level, style):
        """
        Draw a step/ladder barrier line that starts horizontally from y-axis.
        Does NOT draw vertical line from 0 - starts directly at the barrier level.

        Args:
            ax: matplotlib axis
            x_min: start date
            x_max: end date
            schedule: list of {'date': datetime, 'level': float}
            initial_level: initial barrier level
            style: dict with 'color', 'style', 'lw', 'label', 'zorder'
        """
        if not schedule and initial_level is None:
            return

        # Build list of (date, level) changes
        changes = []

        # Start with initial level at x_min
        current_level = initial_level if initial_level else (schedule[0]['level'] if schedule else 100.0)
        changes.append({'date': x_min, 'level': current_level})

        # Add all schedule changes
        for entry in schedule:
            if entry['date'] > x_min and entry['date'] <= x_max:
                changes.append(entry)

        # Sort by date
        changes.sort(key=lambda x: x['date'])

        # Draw horizontal segments and vertical drops (but NOT from 0)
        first_segment = True
        for i, change in enumerate(changes):
            current_date = change['date']
            current_level = change['level']

            # Determine end of this horizontal segment
            if i + 1 < len(changes):
                next_date = changes[i + 1]['date']
            else:
                next_date = x_max

            # Draw horizontal line at current level
            ax.hlines(y=current_level, xmin=current_date, xmax=next_date,
                     color=style['color'], ls=style['style'], lw=style['lw'],
                     label=style['label'] if first_segment else None,
                     zorder=style['zorder'])
            first_segment = False

            # Draw vertical drop to next level (if there is a next level)
            if i + 1 < len(changes):
                next_level = changes[i + 1]['level']
                ax.vlines(x=next_date, ymin=min(current_level, next_level), ymax=max(current_level, next_level),
                         color=style['color'], ls=style['style'], lw=style['lw'],
                         zorder=style['zorder'])

    def plot_professional_scenario(self, scenario_data, meta_data, index=1):
        plt.close('all')

        # Parse dates
        dates_info = meta_data.get('dates', {})
        try:
            start_dt = datetime.strptime(dates_info.get('start', '2024-01-01'), '%Y-%m-%d')
            end_dt = datetime.strptime(dates_info.get('end', '2025-01-01'), '%Y-%m-%d')
        except:
            start_dt = datetime.now()
            end_dt = start_dt + timedelta(days=365)

        # Get calendar and barriers
        calendar = meta_data.get('calendar', [])
        barrier_hierarchy = meta_data.get('barrier_hierarchy', [])
        barrier_levels = self._extract_barrier_levels(barrier_hierarchy)

        # Generate discrete scenario points for ALL observations
        scenario_points, total_obs_count, termination_obs_index = generate_discrete_scenario_points(
            scenario_data, calendar, barrier_hierarchy, start_dt, end_dt
        )

        # Separate points: before/at termination vs after termination
        points_with_price = [pt for pt in scenario_points if pt['price'] is not None]
        all_obs_points = [pt for pt in scenario_points if pt['obs_index'] > 0]

        # Extract data for plotting (only points with prices)
        plot_dates = [pt['date'] for pt in points_with_price]
        plot_prices = [pt['price'] for pt in points_with_price]
        plot_obs_indices = [pt['obs_index'] for pt in points_with_price]

        # Find termination point
        termination_point = None
        for pt in scenario_points:
            if pt.get('is_termination') and pt['price'] is not None:
                termination_point = pt
                break

        # Create figure
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.25,
                              left=0.06, right=0.94, top=0.88, bottom=0.15)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # X-AXIS RANGE: ALWAYS FROM START TO END (same for all scenarios)
        x_min = start_dt
        x_max = end_dt

        # ============================================================
        # PLOT KNOCK-OUT BARRIER (GREEN) - LADDER/STEP SHAPE
        # Using custom method to avoid vertical line from 0
        # ============================================================
        ko_schedule = []
        for entry in calendar:
            try:
                d = datetime.strptime(entry.get('date', ''), '%Y-%m-%d')
                levels = entry.get('levels', {})
                for k, v in levels.items():
                    if 'knock-out' in k.lower() or 'autocall' in k.lower():
                        try:
                            val = float(re.sub(r'[^\d.\-]', '', str(v)))
                            val = self._normalize_value(val)
                            if val is not None:
                                ko_schedule.append({'date': d, 'level': val})
                        except:
                            pass
            except:
                continue

        ko_schedule.sort(key=lambda x: x['date'])

        initial_ko = barrier_levels.get('Knock-out')
        if not initial_ko and ko_schedule:
            initial_ko = ko_schedule[0]['level']
        if not initial_ko:
            initial_ko = 100.0

        # Use custom step drawing method
        self._draw_step_barrier(
            ax1, x_min, x_max, ko_schedule, initial_ko,
            self.BARRIER_CONFIG['Knock-out']
        )

        # ============================================================
        # PLOT KNOCK-IN BARRIER (RED) - HORIZONTAL LINE
        # ============================================================
        ki_val = barrier_levels.get('Knock-in')
        if ki_val is not None and ki_val > 0:
            style = self.BARRIER_CONFIG['Knock-in']
            ax1.hlines(y=ki_val, xmin=x_min, xmax=x_max,
                      color=style['color'], ls=style['style'], lw=style['lw'],
                      label=style['label'], zorder=style['zorder'])
            ax1.fill_between([x_min, x_max], 0, ki_val,
                           color='red', alpha=0.05, zorder=1)

        # ============================================================
        # PLOT COUPON BARRIER (BLUE)
        # ============================================================
        cpn_val = barrier_levels.get('Coupon')
        if cpn_val is not None and cpn_val > 0 and cpn_val != ki_val:
            style = self.BARRIER_CONFIG['Coupon']
            ax1.hlines(y=cpn_val, xmin=x_min, xmax=x_max,
                      color=style['color'], ls=style['style'], lw=style['lw'],
                      label=style['label'], zorder=style['zorder'])

        # ============================================================
        # PLOT STRIKE LINE (100%)
        # ============================================================
        ax1.hlines(y=100, xmin=x_min, xmax=x_max,
                  color=self.BARRIER_CONFIG['Strike']['color'],
                  ls=self.BARRIER_CONFIG['Strike']['style'],
                  lw=self.BARRIER_CONFIG['Strike']['lw'],
                  label='Strike (100%)', zorder=5, alpha=0.7)

        # ============================================================
        # PLOT ASSET PRICE (Points + Line) - ONLY UP TO TERMINATION
        # ============================================================
        if len(plot_dates) > 1:
            ax1.plot(plot_dates, plot_prices, color='#1a1a1a', lw=1.5, zorder=9, alpha=0.8)

        # Plot regular observation points (excluding start and termination)
        regular_points_dates = []
        regular_points_prices = []
        for pt in points_with_price:
            if pt['obs_index'] > 0 and not pt.get('is_termination'):
                regular_points_dates.append(pt['date'])
                regular_points_prices.append(pt['price'])

        if regular_points_dates:
            ax1.scatter(regular_points_dates, regular_points_prices,
                       color='#1a1a1a', s=35, zorder=10,
                       edgecolors='white', linewidths=0.5,
                       label='Asset Price')

        # ============================================================
        # TERMINATION HANDLING
        # ============================================================
        if termination_point:
            term_date = termination_point['date']
            term_price = termination_point['price']
            term_reason = termination_point.get('termination_reason', 'Termination')

            # Draw vertical termination line
            ax1.axvline(x=term_date, color='#FF6B00', ls='-', lw=2.5, zorder=15, alpha=0.9)

            # Mark termination point
            ax1.scatter([term_date], [term_price],
                       color='#FF6B00', s=150, zorder=20,
                       marker='X', edgecolors='white', linewidths=2,
                       label='Termination Point')

            # Add termination label
            label_y = self.Y_AXIS_MAX - 5

            ax1.annotate(term_reason,
                        xy=(term_date, label_y),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='#FF6B00',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#FF6B00', lw=1.5),
                        ha='left', va='top', zorder=25)

            # Annotate final price
            ax1.annotate(f"{term_price:.1f}%",
                        xy=(term_date, term_price),
                        xytext=(-35, -15), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#1a1a1a', alpha=0.9),
                        arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1),
                        zorder=21)

        # ============================================================
        # X-AXIS: SHOW ALL OBSERVATION DATES
        # ============================================================
        tick_dates = []
        tick_labels = []

        # Add start
        tick_dates.append(start_dt)
        tick_labels.append(f"{start_dt.strftime('%m/%d/%y')}\n(Start)")

        # Add ALL observation points (limit display to avoid crowding)
        if len(all_obs_points) <= 12:
            display_obs = all_obs_points
        else:
            indices = list(np.linspace(0, len(all_obs_points)-1, 10, dtype=int))

            if termination_obs_index and termination_obs_index <= len(all_obs_points):
                term_list_idx = termination_obs_index - 1
                if term_list_idx not in indices:
                    indices.append(term_list_idx)
                    indices = sorted(set(indices))

            display_obs = [all_obs_points[i] for i in indices if i < len(all_obs_points)]

        for pt in display_obs:
            actual_idx = pt['obs_index']
            d = pt['date']
            is_term = pt.get('is_termination', False)
            is_after = pt.get('is_after_termination', False)

            tick_dates.append(d)

            if is_term:
                term_type = pt.get('termination_reason', 'End')
                if 'Autocall' in str(term_type):
                    short_reason = 'Autocall'
                elif 'KI Loss' in str(term_type):
                    short_reason = 'KI Loss'
                elif 'KI Recovery' in str(term_type):
                    short_reason = 'Recovery'
                elif 'Safe' in str(term_type):
                    short_reason = 'Safe'
                else:
                    short_reason = 'End'
                tick_labels.append(f"{d.strftime('%m/%d/%y')}\n★ {short_reason}\n(Obs {actual_idx})")
            elif is_after:
                tick_labels.append(f"{d.strftime('%m/%d/%y')}\n(Obs {actual_idx})")
            else:
                tick_labels.append(f"{d.strftime('%m/%d/%y')}\n(Obs {actual_idx})")

        ax1.set_xticks(tick_dates)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)

        # Gray out labels after termination
        if termination_obs_index:
            for i, label in enumerate(ax1.get_xticklabels()):
                if i > 0 and i <= len(display_obs):
                    pt = display_obs[i-1] if i-1 < len(display_obs) else None
                    if pt and pt.get('is_after_termination'):
                        label.set_color('gray')
                        label.set_alpha(0.5)

        # Set x-axis limits to FULL range
        ax1.set_xlim(x_min, x_max)

        # ============================================================
        # Y-AXIS: STANDARDIZED RANGE (0 to 110)
        # ============================================================
        ax1.set_ylim(self.Y_AXIS_MIN, self.Y_AXIS_MAX)

        # ============================================================
        # LEGEND
        # ============================================================
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(),
                   loc='lower center', bbox_to_anchor=(0.5, 1.02),
                   ncol=5, frameon=False, fontsize=8)

        # ============================================================
        # TITLE
        # ============================================================
        raw_title = f"SCENARIO {index}: {scenario_data.get('condition', '')}"
        wrapped_title = "\n".join(textwrap.wrap(raw_title, width=80))
        fig.suptitle(wrapped_title, fontsize=12, fontweight='bold', y=0.98)

        ax1.set_ylabel("Price Level (%)", fontsize=10)
        ax1.grid(True, ls=':', alpha=0.4, zorder=0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # ============================================================
        # ANALYSIS BOX
        # ============================================================
        ax2.axis('off')

        term_obs_display = termination_point.get('obs_index', 'N/A') if termination_point else 'N/A'
        term_date_str = termination_point['date'].strftime('%Y-%m-%d') if termination_point else 'N/A'
        final_price_str = f"{plot_prices[-1]:.1f}%" if plot_prices else "N/A"

        analysis_content = (
            f"SCENARIO ANALYSIS:\n"
            f"• Condition: {scenario_data.get('condition')}\n"
            f"• Outcome: {scenario_data.get('outcome', 'N/A')}\n"
            f"• Termination: {termination_point.get('termination_reason', 'N/A') if termination_point else 'N/A'}\n"
            f"• Termination Date: {term_date_str} (Observation {term_obs_display} of {total_obs_count})\n"
            f"• Final Price: {final_price_str}"
        )

        wrapped_lines = []
        for line in analysis_content.split('\n'):
            wrapped_lines.append("\n".join(textwrap.wrap(line, width=110)))
        wrapped_analysis = "\n".join(wrapped_lines)

        ax2.text(0.02, 0.95, wrapped_analysis, fontsize=9, va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.5', fc='#FAFAFA', ec='#DDDDDD', lw=1),
                 transform=ax2.transAxes, family='monospace')

        fname = f"scenario_{index}_{np.random.randint(10000, 99999)}.png"
        plt.savefig(fname, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close()
        return fname

from dataclasses import dataclass, field

from scipy.stats import norm

PRICING_METHODOLOGY_PROMPT = """You are a quantitative analyst specializing in structured products pricing.

Analyze the following product information and determine the appropriate Monte Carlo pricing methodology.

## Product Information:
**Key Terms:**
{key_terms}

**Dates Schedule:**
{dates_info}

**Product Classification (from Task 2):**
{product_classification}

**Structural Tags:**
{structural_tags}

## Your Task:
Based on the product information, provide a JSON response with the following structure:

{{
    "product_category": "<one of: AUTOCALLABLE, CAPITAL_PROTECTED, REVERSE_CONVERTIBLE, PARTICIPATION, RANGE_ACCRUAL, FIXED_COUPON, ACCUMULATOR, TWIN_WIN, BONUS_CERTIFICATE, DISCOUNT_CERTIFICATE, OTHER>",

    "pricing_features": {{
        "has_autocall": <true/false - product can be called early if underlying hits trigger>,
        "has_knock_in_barrier": <true/false - there is a knock-in/protection barrier>,
        "has_knock_out_barrier": <true/false - there is a knock-out/autocall barrier>,
        "has_coupon_barrier": <true/false - coupons are conditional on barrier>,
        "has_fixed_coupon": <true/false - coupons are guaranteed regardless of underlying>,
        "has_participation": <true/false - upside participation in underlying performance>,
        "has_capital_protection": <true/false - principal is protected at maturity>,
        "has_memory_feature": <true/false - missed coupons can be recovered>,
        "is_worst_of": <true/false - payoff based on worst performing underlying>,
        "has_step_down": <true/false - barriers decrease over time>,
        "has_range_condition": <true/false - payoff depends on staying in a range>,
        "has_cap": <true/false - upside is capped>,
        "has_floor": <true/false - downside is floored>
    }},

    "payoff_at_maturity": {{
        "description": "<describe what happens at maturity in plain English>",
        "formula_type": "<one of: PRINCIPAL_PLUS_COUPON, UNDERLYING_PERFORMANCE, MIN_PRINCIPAL_OR_UNDERLYING, PARTICIPATION_WITH_PROTECTION, RANGE_ACCRUAL, DIGITAL, OTHER>",
        "loss_condition": "<when does investor lose money? e.g., 'underlying below 60% at maturity' or 'never' for capital protected>"
    }},

    "early_redemption": {{
        "can_redeem_early": <true/false>,
        "trigger_condition": "<e.g., 'underlying >= 100% on observation date' or 'N/A'>",
        "redemption_amount": "<e.g., 'principal + accrued coupons' or 'N/A'>"
    }},

    "coupon_structure": {{
        "type": "<one of: FIXED, CONDITIONAL, MEMORY, ACCRUAL, NONE>",
        "payment_condition": "<e.g., 'underlying >= 70%' or 'always' or 'N/A'>",
        "annual_rate": <extract the annual coupon rate as decimal, e.g., 0.08 for 8%>,
        "frequency": "<monthly/quarterly/semi-annual/annual>"
    }},

    "barriers": {{
        "knock_out_level": <decimal or null, e.g., 1.0 for 100%>,
        "knock_in_level": <decimal or null, e.g., 0.6 for 60%>,
        "coupon_barrier_level": <decimal or null>,
        "protection_level": <decimal or null, for capital protected products>
    }},

    "simulation_approach": {{
        "monitoring_type": "<CONTINUOUS for daily monitoring, DISCRETE for observation dates only>",
        "key_dates_to_simulate": "<OBSERVATION_DATES, DAILY, MATURITY_ONLY>",
        "payoff_calculation_notes": "<any special notes for the Monte Carlo engine>"
    }},

    "risk_profile": {{
        "max_loss_potential": "<e.g., '100% of principal', '40% of principal', '0% (capital protected)'>",
        "main_risk_factors": ["<list main risks like 'underlying decline', 'volatility', etc.>"]
    }}
}}

IMPORTANT RULES:
1. If there are NO barriers mentioned, set has_knock_in_barrier and has_knock_out_barrier to false
2. For capital protected products, set has_capital_protection to true and barriers to null
3. For simple participation notes, focus on participation_rate and cap/floor
4. Extract EXACT coupon rates from the terms - convert monthly to annual if needed (monthly * 12)
5. If product is a simple bond/fixed income, set product_category to FIXED_COUPON
6. For products without early redemption, set can_redeem_early to false

Respond ONLY with the JSON object, no other text.
"""

class PricingMethodologyAnalyzer:
    """Uses LLM to determine appropriate pricing methodology."""

    def __init__(self, config):
        self.config = config

    def analyze(self, key_df, dates_df, viz_data, extra_tags) -> Dict:
        """Analyze product and return pricing methodology."""

        # Format key terms
        key_terms_str = ""
        if key_df is not None and not key_df.empty:
            for _, row in key_df.iterrows():
                key_terms_str += f"- {row.get('Term', '')}: {row.get('Value', '')}\n"

        # Format dates
        dates_str = ""
        if dates_df is not None and not dates_df.empty:
            for _, row in dates_df.iterrows():
                dates_str += f"- {row.get('Date Type', '')}: {row.get('Dates', row.get('Date', ''))}\n"

        # Format classification
        classification_str = ""
        if viz_data:
            prod_class = viz_data.get('product_classification', {})
            classification_str = f"Type: {prod_class.get('type', 'Unknown')}\n"
            classification_str += f"Reasoning: {prod_class.get('reasoning', 'N/A')}\n"
            classification_str += f"Core Structure: {viz_data.get('core_structure', 'N/A')}"

        # Format tags
        tags_str = json.dumps(extra_tags, indent=2) if extra_tags else "{}"

        # Build prompt
        prompt = PRICING_METHODOLOGY_PROMPT.format(
            key_terms=key_terms_str or "No key terms extracted",
            dates_info=dates_str or "No dates extracted",
            product_classification=classification_str or "No classification available",
            structural_tags=tags_str
        )

        try:
            response = self.config.client.messages.create(
                model=self.config.model,
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Clean JSON
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)

            methodology = json.loads(response_text)
            return methodology

        except Exception as e:
            print(f"  Warning: Pricing methodology analysis failed: {e}")
            # Return default autocallable methodology
            return self._get_default_methodology()

    def _get_default_methodology(self) -> Dict:
        """Return default methodology for fallback."""
        return {
            "product_category": "AUTOCALLABLE",
            "pricing_features": {
                "has_autocall": True,
                "has_knock_in_barrier": True,
                "has_knock_out_barrier": True,
                "has_coupon_barrier": True,
                "has_fixed_coupon": False,
                "has_participation": False,
                "has_capital_protection": False,
                "has_memory_feature": False,
                "is_worst_of": False,
                "has_step_down": False,
                "has_range_condition": False,
                "has_cap": False,
                "has_floor": False
            },
            "payoff_at_maturity": {
                "description": "Principal returned if no knock-in, otherwise linked to underlying",
                "formula_type": "MIN_PRINCIPAL_OR_UNDERLYING",
                "loss_condition": "underlying below knock-in at any time AND below 100% at maturity"
            },
            "early_redemption": {
                "can_redeem_early": True,
                "trigger_condition": "underlying >= 100% on observation date",
                "redemption_amount": "principal + accrued coupons"
            },
            "coupon_structure": {
                "type": "CONDITIONAL",
                "payment_condition": "underlying >= coupon barrier",
                "annual_rate": 0.06,
                "frequency": "monthly"
            },
            "barriers": {
                "knock_out_level": 1.0,
                "knock_in_level": 0.6,
                "coupon_barrier_level": 0.7,
                "protection_level": None
            },
            "simulation_approach": {
                "monitoring_type": "CONTINUOUS",
                "key_dates_to_simulate": "OBSERVATION_DATES",
                "payoff_calculation_notes": "Standard autocallable with barrier monitoring"
            },
            "risk_profile": {
                "max_loss_potential": "100% of principal",
                "main_risk_factors": ["underlying decline", "knock-in breach"]
            }
        }

class PricingParameters:
    """Pricing parameters for all product types."""
    # Basic info
    notional: float = 1.0  # <--- CHANGED THIS TO 1.0
    currency: str = "USD"
    product_category: str = "AUTOCALLABLE"

    # Underlyings
    n_underlyings: int = 1
    underlying_names: List[str] = field(default_factory=list)
    initial_levels: np.ndarray = field(default_factory=lambda: np.array([100.0]))

    # Barriers (None if not applicable)
    ko_barrier: float = None
    ki_barrier: float = None
    coupon_barrier: float = None
    protection_level: float = None
    ko_schedule: List[Tuple[int, float]] = field(default_factory=list)

    # Coupon
    coupon_rate: float = 0.0
    coupon_type: str = "CONDITIONAL"  # FIXED, CONDITIONAL, MEMORY, ACCRUAL, NONE
    coupon_frequency: str = "monthly"

    # Product features (from prompt engineering)
    has_autocall: bool = False
    has_knock_in: bool = False
    has_knock_out: bool = False
    has_coupon_barrier: bool = False
    has_fixed_coupon: bool = False
    has_participation: bool = False
    has_capital_protection: bool = False
    has_memory_feature: bool = False
    is_worst_of: bool = False
    has_step_down: bool = False
    has_range_condition: bool = False
    has_cap: bool = False
    has_floor: bool = False

    # Participation
    participation_rate: float = 1.0
    cap_level: float = None
    floor_level: float = None

    # Range (for range accrual)
    range_lower: float = None
    range_upper: float = None

    # Market assumptions
    risk_free_rate: float = 0.03
    dividend_yields: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    volatilities: np.ndarray = field(default_factory=lambda: np.array([0.25]))
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))

    # Timing
    maturity_years: float = 1.0
    observation_dates: List[str] = field(default_factory=list)

    # Simulation
    n_simulations: int = 50000
    n_steps_per_year: int = 252
    monitoring_type: str = "CONTINUOUS"

    # Raw values for validation
    raw_coupon_rate: str = ""
    raw_ko_barrier: str = ""
    raw_ki_barrier: str = ""

    # Methodology from LLM
    pricing_methodology: Dict = field(default_factory=dict)

class PricingParameterExtractor:
    """Extract parameters using both rule-based and LLM analysis."""

    def __init__(self, config=None):
        self.config = config
        self.params = PricingParameters()
        self.extraction_log = []

    def extract_all(self, key_df, dates_df, viz_data, extra_tags) -> PricingParameters:
        """Extract all parameters using LLM methodology + rule-based extraction."""

        # Step 1: Get pricing methodology from LLM
        if self.config:
            analyzer = PricingMethodologyAnalyzer(self.config)
            methodology = analyzer.analyze(key_df, dates_df, viz_data, extra_tags)
            self.params.pricing_methodology = methodology
            self._apply_methodology(methodology)

        # Step 2: Rule-based extraction (can override/supplement LLM)
        self._extract_from_key_terms(key_df)
        self._extract_from_dates(dates_df)
        self._extract_from_viz_data(viz_data)
        self._extract_from_tags(extra_tags)

        # Step 3: Finalize
        self._finalize_params()

        return self.params

    def _apply_methodology(self, methodology: Dict) -> None:
        """Apply LLM-determined methodology to parameters."""

        # Product category
        self.params.product_category = methodology.get('product_category', 'AUTOCALLABLE')

        # Pricing features
        features = methodology.get('pricing_features', {})
        self.params.has_autocall = features.get('has_autocall', False)
        self.params.has_knock_in = features.get('has_knock_in_barrier', False)
        self.params.has_knock_out = features.get('has_knock_out_barrier', False)
        self.params.has_coupon_barrier = features.get('has_coupon_barrier', False)
        self.params.has_fixed_coupon = features.get('has_fixed_coupon', False)
        self.params.has_participation = features.get('has_participation', False)
        self.params.has_capital_protection = features.get('has_capital_protection', False)
        self.params.has_memory_feature = features.get('has_memory_feature', False)
        self.params.is_worst_of = features.get('is_worst_of', False)
        self.params.has_step_down = features.get('has_step_down', False)
        self.params.has_range_condition = features.get('has_range_condition', False)
        self.params.has_cap = features.get('has_cap', False)
        self.params.has_floor = features.get('has_floor', False)

        # Barriers from methodology
        barriers = methodology.get('barriers', {})
        if barriers.get('knock_out_level') is not None:
            self.params.ko_barrier = barriers['knock_out_level']
        if barriers.get('knock_in_level') is not None:
            self.params.ki_barrier = barriers['knock_in_level']
        if barriers.get('coupon_barrier_level') is not None:
            self.params.coupon_barrier = barriers['coupon_barrier_level']
        if barriers.get('protection_level') is not None:
            self.params.protection_level = barriers['protection_level']

        # Coupon from methodology
        coupon = methodology.get('coupon_structure', {})
        self.params.coupon_type = coupon.get('type', 'CONDITIONAL')
        if coupon.get('annual_rate') is not None:
            self.params.coupon_rate = coupon['annual_rate']
        if coupon.get('frequency'):
            self.params.coupon_frequency = coupon['frequency']

        # Simulation approach
        sim = methodology.get('simulation_approach', {})
        self.params.monitoring_type = sim.get('monitoring_type', 'CONTINUOUS')

    def _extract_from_key_terms(self, key_df) -> None:
        """Extract from Task 1 key terms (supplements/overrides LLM)."""
        if key_df is None or key_df.empty:
            return

        for _, row in key_df.iterrows():
            term = str(row.get('Term', '')).strip()
            term_lower = term.lower()
            value = str(row.get('Value', '')).strip()

            # Notional
            if any(kw in term_lower for kw in ['principal', 'notional', 'nominal', 'denomination']):
                parsed = self._parse_number(value)
                if 0:
                    self.params.notional = parsed

            # Currency
            if 'currency' in term_lower:
                self.params.currency = self._extract_currency(value)

            # Underlyings
            if 'underlying' in term_lower:
                parts = re.split(r'[,;/&]|\band\b', value)
                names = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
                if names:
                    self.params.n_underlyings = len(names)
                    self.params.underlying_names = names
                    self.params.initial_levels = np.array([100.0] * len(names))
                    self.params.volatilities = np.array([0.25] * len(names))
                    self.params.dividend_yields = np.array([0.0] * len(names))

            # KO Barrier - only if product has autocall
            if self.params.has_autocall or self.params.has_knock_out:
                if any(kw in term_lower for kw in ['autocall', 'knock-out', 'call trigger', 'early redemption trigger']):
                    pcts = self._extract_percentages(value)
                    if pcts:
                        self.params.ko_barrier = pcts[0]
                        self.params.raw_ko_barrier = value

            # KI Barrier - only if product has knock-in
            if self.params.has_knock_in:
                if any(kw in term_lower for kw in ['knock-in', 'kick-in', 'barrier level', 'protection barrier']):
                    if 'coupon' not in term_lower:
                        pcts = self._extract_percentages(value)
                        if pcts:
                            self.params.ki_barrier = pcts[0]
                            self.params.raw_ki_barrier = value

            # Coupon barrier
            if self.params.has_coupon_barrier:
                if 'coupon barrier' in term_lower or 'distribution barrier' in term_lower:
                    pcts = self._extract_percentages(value)
                    if pcts:
                        self.params.coupon_barrier = pcts[0]

            # Coupon rate
            if any(kw in term_lower for kw in ['coupon', 'interest', 'yield']) and 'barrier' not in term_lower:
                rate = self._parse_coupon_rate(value)
                if rate is not None:
                    self.params.coupon_rate = rate
                    self.params.raw_coupon_rate = value

            # Participation
            if 'participation' in term_lower:
                pcts = self._extract_percentages(value)
                if pcts:
                    self.params.participation_rate = pcts[0]
                    self.params.has_participation = True

            # Cap
            if 'cap' in term_lower and 'floor' not in term_lower:
                pcts = self._extract_percentages(value)
                if pcts:
                    self.params.cap_level = pcts[0]
                    self.params.has_cap = True

            # Floor
            if 'floor' in term_lower or 'minimum return' in term_lower:
                pcts = self._extract_percentages(value)
                if pcts:
                    self.params.floor_level = pcts[0]
                    self.params.has_floor = True

            # Protection level (capital protected)
            if 'protection' in term_lower and 'barrier' not in term_lower:
                pcts = self._extract_percentages(value)
                if pcts:
                    self.params.protection_level = pcts[0]
                    self.params.has_capital_protection = True

    def _extract_from_dates(self, dates_df) -> None:
        """Extract dates from Task 1."""
        if dates_df is None or dates_df.empty:
            return

        issue_date = None
        maturity_date = None

        for _, row in dates_df.iterrows():
            date_type = str(row.get('Date Type', '')).lower()
            date_val = str(row.get('Dates', row.get('Date', '')))
            trigger = str(row.get('Trigger / Barrier Level', ''))

            explicit_dates = self._extract_explicit_dates(date_val)

            if 'issue' in date_type and 'maturity' not in date_type:
                if explicit_dates:
                    issue_date = explicit_dates[0]

            elif 'maturity' in date_type or 'final' in date_type:
                if explicit_dates:
                    maturity_date = explicit_dates[-1]

            elif 'observation' in date_type or 'valuation' in date_type:
                barriers = self._extract_percentages(trigger)
                for i, d in enumerate(explicit_dates):
                    self.params.observation_dates.append(d.strftime('%Y-%m-%d'))
                    if self.params.has_autocall and issue_date:
                        barrier = barriers[i] if i < len(barriers) else (self.params.ko_barrier or 1.0)
                        days = (d - issue_date).days
                        if days > 0:
                            self.params.ko_schedule.append((days, barrier))

        if issue_date and maturity_date:
            self.params.maturity_years = (maturity_date - issue_date).days / 365.0

        if self.params.ko_schedule:
            self.params.ko_schedule.sort(key=lambda x: x[0])

    def _extract_from_viz_data(self, viz_data: Dict) -> None:
        """Extract from Task 2 analysis."""
        if not viz_data:
            return

        # Only fill in gaps, don't override
        if self.params.maturity_years == 1.0:
            dates_info = viz_data.get('dates', {})
            if dates_info:
                try:
                    start = datetime.strptime(dates_info.get('start', ''), '%Y-%m-%d')
                    end = datetime.strptime(dates_info.get('end', ''), '%Y-%m-%d')
                    self.params.maturity_years = (end - start).days / 365.0
                except:
                    pass

    def _extract_from_tags(self, tags: Dict) -> None:
        """Extract from structural tags."""
        if not tags:
            return

        if tags.get('is_memory'):
            self.params.has_memory_feature = True
        if tags.get('is_worst_of'):
            self.params.is_worst_of = True

    def _finalize_params(self) -> None:
        """Finalize parameters."""
        n = self.params.n_underlyings
        if n > 1:
            corr = np.full((n, n), 0.5)
            np.fill_diagonal(corr, 1.0)
            self.params.correlation_matrix = corr
        else:
            self.params.correlation_matrix = np.array([[1.0]])

    def _extract_explicit_dates(self, text: str) -> List[datetime]:
        """Extract only explicit calendar dates."""
        dates = []
        text = text.replace('<br>', ' ').replace('\n', ' ')

        # YYYY-MM-DD
        for m in re.finditer(r'(\d{4})-(\d{2})-(\d{2})', text):
            try:
                dates.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except:
                pass

        # Month DD, YYYY
        for m in re.finditer(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', text, re.I):
            try:
                month = datetime.strptime(m.group(1), '%B').month
                dates.append(datetime(int(m.group(3)), month, int(m.group(2))))
            except:
                pass

        # DD Month YYYY
        for m in re.finditer(r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', text, re.I):
            try:
                month = datetime.strptime(m.group(2), '%B').month
                dates.append(datetime(int(m.group(3)), month, int(m.group(1))))
            except:
                pass

        seen = set()
        unique = []
        for d in dates:
            key = d.strftime('%Y-%m-%d')
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return sorted(unique)

    def _extract_percentages(self, text: str) -> List[float]:
        """Extract percentages as decimals."""
        pcts = []
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text):
            val = float(m.group(1))
            pcts.append(val / 100.0 if val > 1 else val)
        return pcts

    def _parse_coupon_rate(self, text: str) -> Optional[float]:
        """Parse coupon rate, handling per-period rates."""
        text_lower = text.lower()

        # Per period
        match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:per|each)?\s*(month|quarter|year|annum|p\.?a\.?)', text_lower)
        if match:
            rate = float(match.group(1))
            rate = rate / 100.0 if rate > 1 else rate
            period = match.group(2)
            if 'month' in period:
                rate *= 12
            elif 'quarter' in period:
                rate *= 4
            return rate

        # Simple percentage
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if match:
            rate = float(match.group(1))
            rate = rate / 100.0 if rate > 1 else rate
            if rate < 0.03 and 'month' in text_lower:
                rate *= 12
            return rate
        return None

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse number from text."""
        try:
            clean = re.sub(r'[^\d.\-]', '', text.replace(',', ''))
            return float(clean) if clean else None
        except:
            return None

    def _extract_currency(self, text: str) -> str:
        """Extract currency code."""
        for c in ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'HKD', 'SGD', 'AUD', 'CHF', 'CAD']:
            if c in text.upper():
                return c
        return 'USD'

class UniversalMonteCarloPricer:
    """Monte Carlo pricer that adapts to any product type."""

    def __init__(self, params: PricingParameters):
        self.params = params
        self.paths = None
        self.payoffs = None
        self.statistics = {}

    def _generate_paths(self):
        """Generate correlated price paths."""
        n_sims = self.params.n_simulations
        n_steps = int(self.params.maturity_years * self.params.n_steps_per_year)
        n_assets = self.params.n_underlyings
        dt = 1.0 / self.params.n_steps_per_year

        try:
            L = np.linalg.cholesky(self.params.correlation_matrix)
        except:
            L = np.eye(n_assets)

        Z = np.random.standard_normal((n_sims, n_steps, n_assets))
        Z_corr = np.einsum('ijk,lk->ijl', Z, L)

        paths = np.zeros((n_sims, n_steps + 1, n_assets))
        paths[:, 0, :] = self.params.initial_levels

        for i in range(n_assets):
            vol = self.params.volatilities[i]
            r = self.params.risk_free_rate
            q = self.params.dividend_yields[i]
            drift = (r - q - 0.5 * vol**2) * dt
            diffusion = vol * np.sqrt(dt)

            for t in range(n_steps):
                paths[:, t+1, i] = paths[:, t, i] * np.exp(drift + diffusion * Z_corr[:, t, i])

        self.paths = paths
        return paths

    def _get_performance(self):
        """Get performance measure (worst-of or single)."""
        normalized = self.paths / self.params.initial_levels
        if self.params.is_worst_of and self.params.n_underlyings > 1:
            return np.min(normalized, axis=2)
        return normalized[:, :, 0]

    def _calculate_payoffs(self):
        """Calculate payoffs based on product category."""
        category = self.params.product_category

        if category == 'CAPITAL_PROTECTED':
            return self._payoff_capital_protected()
        elif category == 'PARTICIPATION':
            return self._payoff_participation()
        elif category == 'FIXED_COUPON':
            return self._payoff_fixed_coupon()
        elif category == 'REVERSE_CONVERTIBLE':
            return self._payoff_reverse_convertible()
        elif category == 'RANGE_ACCRUAL':
            return self._payoff_range_accrual()
        elif category == 'BONUS_CERTIFICATE':
            return self._payoff_bonus_certificate()
        elif category == 'DISCOUNT_CERTIFICATE':
            return self._payoff_discount_certificate()
        else:
            # Default: Autocallable (covers Phoenix, Snowball, etc.)
            return self._payoff_autocallable()

    def _payoff_autocallable(self):
        """Standard autocallable with optional barriers."""
        n_sims = self.params.n_simulations
        n_steps = self.paths.shape[1] - 1
        performance = self._get_performance()

        payoffs = np.zeros(n_sims)
        autocalled = np.zeros(n_sims, dtype=bool)
        ki_triggered = np.zeros(n_sims, dtype=bool)
        coupons = np.zeros(n_sims)
        missed_coupons = np.zeros(n_sims)
        autocall_times = np.zeros(n_sims)

        # Observation schedule
        if self.params.ko_schedule:
            obs_days = [d for d, _ in self.params.ko_schedule]
        else:
            freq_map = {'monthly': 21, 'quarterly': 63, 'semi-annual': 126, 'annual': 252}
            interval = freq_map.get(self.params.coupon_frequency, 21)
            obs_days = list(range(interval, n_steps + 1, interval))

        obs_steps = [min(int(d * self.params.n_steps_per_year / 365), n_steps) for d in obs_days]
        obs_steps = [s for s in obs_steps if 0 < s <= n_steps]
        n_periods = max(len(obs_steps), 1)
        coupon_per_period = self.params.coupon_rate / n_periods

        # Check knock-in (if applicable)
        if self.params.has_knock_in and self.params.ki_barrier is not None:
            for sim in range(n_sims):
                if np.min(performance[sim, :]) <= self.params.ki_barrier:
                    ki_triggered[sim] = True

        # Process observations
        for obs_idx, step in enumerate(obs_steps):
            day = int(step * 365 / self.params.n_steps_per_year)

            # Get applicable KO barrier
            ko_barrier = self.params.ko_barrier or 1.0
            if self.params.ko_schedule:
                for obs_day, obs_barrier in self.params.ko_schedule:
                    if day >= obs_day:
                        ko_barrier = obs_barrier

            for sim in range(n_sims):
                if autocalled[sim]:
                    continue

                perf = performance[sim, step]

                # Autocall check
                if self.params.has_autocall and perf >= ko_barrier:
                    autocalled[sim] = True
                    autocall_times[sim] = obs_idx + 1
                    time_frac = (obs_idx + 1) / n_periods
                    disc = np.exp(-self.params.risk_free_rate * time_frac * self.params.maturity_years)
                    total_coupon = coupons[sim] + coupon_per_period
                    if self.params.has_memory_feature:
                        total_coupon += missed_coupons[sim]
                    payoffs[sim] = (1.0 + total_coupon) * disc

                # Coupon check
                elif self.params.has_fixed_coupon:
                    coupons[sim] += coupon_per_period
                elif self.params.has_coupon_barrier and self.params.coupon_barrier is not None:
                    if perf >= self.params.coupon_barrier:
                        coupons[sim] += coupon_per_period
                        if self.params.has_memory_feature:
                            coupons[sim] += missed_coupons[sim]
                            missed_coupons[sim] = 0
                    elif self.params.has_memory_feature:
                        missed_coupons[sim] += coupon_per_period
                else:
                    coupons[sim] += coupon_per_period

        # Final redemption
        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        for sim in range(n_sims):
            if autocalled[sim]:
                continue

            final_perf = performance[sim, -1]

            if ki_triggered[sim] and final_perf < 1.0:
                loss = 1.0 - final_perf
                if self.params.has_floor and self.params.floor_level is not None:
                    loss = min(loss, 1.0 - self.params.floor_level)
                payoffs[sim] = (1.0 - loss + coupons[sim]) * disc
            else:
                total_coupon = coupons[sim]
                if self.params.has_memory_feature:
                    total_coupon += missed_coupons[sim]
                payoffs[sim] = (1.0 + total_coupon) * disc

        self.statistics = {
            'prob_autocall': np.mean(autocalled),
            'prob_ki': np.mean(ki_triggered),
            'prob_loss': np.mean((ki_triggered) & (~autocalled) & (performance[:, -1] < 1.0)),
            'avg_coupons': np.mean(coupons),
            'avg_autocall_period': np.mean(autocall_times[autocalled]) if np.any(autocalled) else 0,
            'n_observations': n_periods
        }

        self.payoffs = payoffs * self.params.notional
        return self.payoffs

    def _payoff_capital_protected(self):
        """Capital protected note - principal guaranteed."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]

        protection = self.params.protection_level or 1.0
        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            perf = final_perf[sim]
            base = protection

            # Add coupon if any
            if self.params.coupon_rate > 0:
                base += self.params.coupon_rate * self.params.maturity_years

            # Add participation if applicable
            if self.params.has_participation and perf > 1.0:
                upside = (perf - 1.0) * self.params.participation_rate
                if self.params.has_cap and self.params.cap_level:
                    upside = min(upside, self.params.cap_level - 1.0)
                base += upside

            payoffs[sim] = max(protection, base)

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': 0,
            'avg_coupons': self.params.coupon_rate * self.params.maturity_years,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_participation(self):
        """Participation certificate - track underlying with leverage."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]

        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            perf = final_perf[sim]

            # Apply participation
            if perf > 1.0:
                ret = (perf - 1.0) * self.params.participation_rate
                if self.params.has_cap and self.params.cap_level:
                    ret = min(ret, self.params.cap_level - 1.0)
                payoffs[sim] = 1.0 + ret
            else:
                if self.params.has_floor and self.params.floor_level:
                    payoffs[sim] = max(self.params.floor_level, perf)
                else:
                    payoffs[sim] = perf

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': np.mean(final_perf < 1.0),
            'avg_coupons': 0,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_fixed_coupon(self):
        """Fixed coupon note - like a bond."""
        n_sims = self.params.n_simulations

        total_coupon = self.params.coupon_rate * self.params.maturity_years
        payoff = (1.0 + total_coupon)

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = np.full(n_sims, payoff * self.params.notional * disc)

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': 0,
            'avg_coupons': total_coupon,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_reverse_convertible(self):
        """Reverse convertible - high coupon, downside exposure."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]
        min_perf = np.min(performance, axis=1)

        total_coupon = self.params.coupon_rate * self.params.maturity_years
        payoffs = np.zeros(n_sims)
        ki_triggered = np.zeros(n_sims, dtype=bool)

        if self.params.has_knock_in and self.params.ki_barrier:
            ki_triggered = min_perf <= self.params.ki_barrier

        for sim in range(n_sims):
            if ki_triggered[sim] and final_perf[sim] < 1.0:
                payoffs[sim] = final_perf[sim] + total_coupon
            else:
                payoffs[sim] = 1.0 + total_coupon

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': np.mean(ki_triggered),
            'prob_loss': np.mean((ki_triggered) & (final_perf < 1.0)),
            'avg_coupons': total_coupon,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_range_accrual(self):
        """Range accrual - coupon accrues when in range."""
        n_sims = self.params.n_simulations
        n_steps = self.paths.shape[1] - 1
        performance = self._get_performance()

        lower = self.params.range_lower or 0.8
        upper = self.params.range_upper or 1.2

        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            days_in_range = np.sum((performance[sim, :] >= lower) & (performance[sim, :] <= upper))
            accrual_rate = days_in_range / (n_steps + 1)
            coupon = self.params.coupon_rate * self.params.maturity_years * accrual_rate
            payoffs[sim] = 1.0 + coupon

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': 0,
            'avg_coupons': np.mean(self.payoffs / self.params.notional / disc - 1.0),
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_bonus_certificate(self):
        """Bonus certificate - bonus if barrier not breached."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]
        min_perf = np.min(performance, axis=1)

        barrier = self.params.ki_barrier or 0.7
        bonus = self.params.cap_level or 1.2

        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            if min_perf[sim] > barrier:
                payoffs[sim] = max(bonus, final_perf[sim])
            else:
                payoffs[sim] = final_perf[sim]

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': np.mean(min_perf <= barrier),
            'prob_loss': np.mean(self.payoffs < self.params.notional * disc),
            'avg_coupons': 0,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_discount_certificate(self):
        """Discount certificate - buy at discount, capped upside."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]

        cap = self.params.cap_level or 1.1

        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            payoffs[sim] = min(final_perf[sim], cap)

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': np.mean(final_perf < 1.0),
            'avg_coupons': 0,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def price(self):
        """Run pricing."""
        np.random.seed(42)
        self._generate_paths()
        payoffs = self._calculate_payoffs()

        fair_value = np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(len(payoffs))

        return {
            'fair_value': fair_value,
            'std_error': std_error,
            'confidence_interval_95': (fair_value - 1.96 * std_error, fair_value + 1.96 * std_error),
            'prob_autocall': self.statistics.get('prob_autocall', 0),
            'prob_ki': self.statistics.get('prob_ki', 0),
            'prob_loss': self.statistics.get('prob_loss', 0),
            'expected_return': (fair_value - self.params.notional) / self.params.notional,
            'avg_coupons_received': self.statistics.get('avg_coupons', 0) * self.params.notional,
            'avg_autocall_period': self.statistics.get('avg_autocall_period', 0),
            'n_observations': self.statistics.get('n_observations', 0),
        }

    def calculate_greeks(self):
        """Calculate Greeks."""
        base = self.price()['fair_value']
        bump = 0.01

        # Delta
        orig = self.params.initial_levels.copy()
        self.params.initial_levels = orig * (1 + bump)
        self._generate_paths()
        up = self._calculate_payoffs().mean()
        self.params.initial_levels = orig * (1 - bump)
        self._generate_paths()
        down = self._calculate_payoffs().mean()
        delta = (up - down) / (2 * bump * orig[0])
        self.params.initial_levels = orig

        # Vega
        orig_vol = self.params.volatilities.copy()
        self.params.volatilities = orig_vol + 0.01
        self._generate_paths()
        up = self._calculate_payoffs().mean()
        self.params.volatilities = orig_vol - 0.01
        self._generate_paths()
        down = self._calculate_payoffs().mean()
        vega = (up - down) / 0.02
        self.params.volatilities = orig_vol

        # Theta
        orig_mat = self.params.maturity_years
        self.params.maturity_years = orig_mat - 1/365
        self._generate_paths()
        later = self._calculate_payoffs().mean()
        theta = (later - base) / (1/365)
        self.params.maturity_years = orig_mat

        # Rho
        orig_rate = self.params.risk_free_rate
        self.params.risk_free_rate = orig_rate + 0.0001
        self._generate_paths()
        up = self._calculate_payoffs().mean()
        self.params.risk_free_rate = orig_rate - 0.0001
        self._generate_paths()
        down = self._calculate_payoffs().mean()
        rho = (up - down) / 0.0002
        self.params.risk_free_rate = orig_rate

        self._generate_paths()
        return {'delta': delta, 'vega': vega, 'theta': theta, 'rho': rho}

    def calculate_risk_metrics(self):
        """Calculate risk metrics."""
        if self.payoffs is None:
            self.price()

        payoffs = self.payoffs
        var_95 = self.params.notional - np.percentile(payoffs, 5)
        var_99 = self.params.notional - np.percentile(payoffs, 1)
        es_95 = self.params.notional - np.mean(payoffs[payoffs <= np.percentile(payoffs, 5)])
        max_loss = self.params.notional - np.min(payoffs)

        return {
            'var_95': max(0, var_95),
            'var_99': max(0, var_99),
            'expected_shortfall_95': max(0, es_95),
            'max_loss': max(0, max_loss),
            'probability_of_loss': np.mean(payoffs < self.params.notional),
        }

def generate_pricing_html_report(params, results, greeks, risks):
    """Generate HTML report adapted to product type with Risk & Greek Explanations."""

    expected_return = results.get('expected_return', 0)
    if expected_return > 0.02:
        recommendation, rec_color = "ATTRACTIVE", "#28a745"
    elif expected_return > -0.01:
        recommendation, rec_color = "NEUTRAL", "#ffc107"
    else:
        recommendation, rec_color = "REVIEW REQUIRED", "#dc3545"

    # Product description
    category = params.product_category
    features = []
    if params.is_worst_of:
        features.append("Worst-of")
    if params.has_memory_feature:
        features.append("Memory")
    if params.has_capital_protection:
        features.append("Capital Protected")
    if params.has_participation:
        features.append(f"Participation ({params.participation_rate:.0%})")
    if params.has_autocall:
        features.append("Autocallable")

    product_desc = f"{' '.join(features)} {category.replace('_', ' ').title()}"

    # Methodology from LLM
    methodology = params.pricing_methodology
    payoff_desc = methodology.get('payoff_at_maturity', {}).get('description', 'N/A') if methodology else 'N/A'
    loss_condition = methodology.get('payoff_at_maturity', {}).get('loss_condition', 'N/A') if methodology else 'N/A'

    # Build barriers section (only if applicable)
    barriers_html = ""
    if params.has_knock_out or params.has_knock_in or params.has_coupon_barrier:
        barriers_html = f"""
        <tr>
            <td style="padding: 8px 0; color: #333;">KO Barrier</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.ko_barrier:.2%}" if params.ko_barrier else "N/A"}</td>
            <td style="padding: 8px 0 8px 30px; color: #333;">KI Barrier</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.ki_barrier:.2%}" if params.ki_barrier else "N/A"}</td>
        </tr>
        <tr>
            <td style="padding: 8px 0; color: #333;">Coupon Barrier</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.coupon_barrier:.2%}" if params.coupon_barrier else "N/A"}</td>
            <td style="padding: 8px 0 8px 30px; color: #333;">Protection Level</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.protection_level:.2%}" if params.protection_level else "N/A"}</td>
        </tr>
        """
    elif params.has_capital_protection:
        barriers_html = f"""
        <tr>
            <td style="padding: 8px 0; color: #333;">Protection Level</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.protection_level:.2%}" if params.protection_level else "100%"}</td>
            <td style="padding: 8px 0 8px 30px; color: #333;">Cap Level</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.cap_level:.2%}" if params.cap_level else "N/A"}</td>
        </tr>
        """

    # Probabilities section (only show relevant ones)
    prob_html = ""
    if params.has_autocall:
        prob_html += f'<tr><td style="padding: 10px 0;">Prob. Autocall</td><td style="text-align: right;"><span style="background: #e0e0e0; padding: 3px 10px; border-radius: 10px; font-weight: 600;">{results["prob_autocall"]:.1%}</span></td></tr>'
    if params.has_knock_in:
        prob_html += f'<tr><td style="padding: 10px 0;">Prob. Knock-In</td><td style="text-align: right;"><span style="background: #ffe0b2; padding: 3px 10px; border-radius: 10px; font-weight: 600;">{results["prob_ki"]:.1%}</span></td></tr>'
    if results['prob_loss'] > 0 or not params.has_capital_protection:
        prob_html += f'<tr><td style="padding: 10px 0;">Prob. Loss</td><td style="text-align: right;"><span style="background: #ffcdd2; color: #c00; padding: 3px 10px; border-radius: 10px; font-weight: 600;">{results["prob_loss"]:.1%}</span></td></tr>'

    prob_safe = max(0, 1 - results['prob_autocall'] - results['prob_loss'])
    prob_html += f'<tr><td style="padding: 10px 0;">Prob. Full Principal</td><td style="text-align: right;"><span style="background: #c8e6c9; padding: 3px 10px; border-radius: 10px; font-weight: 600;">{prob_safe:.1%}</span></td></tr>'

    html = f'''
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 950px; margin: 20px auto;">
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #333 100%); color: white; padding: 25px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0; font-size: 24px;">Monte Carlo Pricing Analysis (Task 3)</h2>
            <p style="margin: 8px 0 0; opacity: 0.9; font-size: 14px;">{product_desc}</p>
            <p style="margin: 5px 0 0; opacity: 0.8; font-size: 12px;">
                {params.n_simulations:,} Simulations | {params.n_underlyings} Underlying(s) | {params.maturity_years:.1f} Years
            </p>
        </div>

        <div style="background: #e3f2fd; border: 1px solid #1976d2; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px; color: #1565c0; font-size: 14px;">🤖 AI-Determined Pricing Methodology</h4>
            <p style="margin: 0 0 8px; font-size: 12px;"><b>Product Category:</b> {category}</p>
            <p style="margin: 0 0 8px; font-size: 12px;"><b>Payoff at Maturity:</b> {payoff_desc}</p>
            <p style="margin: 0; font-size: 12px;"><b>Loss Condition:</b> {loss_condition}</p>
        </div>

        <div style="background: #e8f5e9; border: 1px solid #4caf50; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px; color: #2e7d32; font-size: 14px;">✓ Parameters from Task 1</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td style="padding: 4px;"><b>Coupon Rate:</b></td>
                    <td>{params.coupon_rate:.4%} p.a. ({params.coupon_type})</td>
                </tr>
                {f'<tr><td style="padding: 4px;"><b>KO Barrier:</b></td><td>{params.ko_barrier:.2%}</td></tr>' if params.ko_barrier else ''}
                {f'<tr><td style="padding: 4px;"><b>KI Barrier:</b></td><td>{params.ki_barrier:.2%}</td></tr>' if params.ki_barrier else ''}
            </table>
        </div>

        <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 15px; color: #000; font-size: 16px; border-bottom: 2px solid #000; padding-bottom: 8px;">Product Configuration</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <tr>
                    <td style="padding: 8px 0; color: #333; width: 25%;">Notional</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600; width: 25%;">{params.currency} {params.notional:,.2f}</td>
                    <td style="padding: 8px 0 8px 30px; color: #333; width: 25%;">Maturity</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600; width: 25%;">{params.maturity_years:.2f} years</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #333;">Coupon Rate (p.a.)</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600;">{params.coupon_rate:.4%}</td>
                    <td style="padding: 8px 0 8px 30px; color: #333;">Coupon Type</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600;">{params.coupon_type}</td>
                </tr>
                {barriers_html}
                <tr>
                    <td style="padding: 8px 0; color: #333;">Volatility</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600;">{params.volatilities[0]:.1%}</td>
                    <td style="padding: 8px 0 8px 30px; color: #333;">Risk-Free Rate</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600;">{params.risk_free_rate:.2%}</td>
                </tr>
            </table>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                <h3 style="margin: 0 0 15px; color: #000; font-size: 16px; border-bottom: 2px solid #000; padding-bottom: 8px;">Valuation Results</h3>
                <div style="text-align: center; padding: 15px 0; background: #f5f5f5; border-radius: 8px; margin-bottom: 15px;">
                    <div style="font-size: 32px; font-weight: 700; color: #000;">{params.currency} {results['fair_value']:,.2f}</div>
                    <div style="font-size: 12px; color: #333;">Fair Value (Per Unit)</div>
                </div>
                <table style="width: 100%; font-size: 12px;">
                    <tr><td style="padding: 6px 0;">95% CI</td><td style="text-align: right;">{params.currency} {results['confidence_interval_95'][0]:,.2f} - {results['confidence_interval_95'][1]:,.2f}</td></tr>
                    <tr style="background: #f5f5f5;"><td style="padding: 8px 0; font-weight: 600;">Expected Return</td><td style="text-align: right; font-weight: 700; font-size: 14px; color: {'#000' if expected_return >= 0 else '#c00'};">{expected_return:+.2%}</td></tr>
                </table>
            </div>

            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                <h3 style="margin: 0 0 15px; color: #000; font-size: 16px; border-bottom: 2px solid #000; padding-bottom: 8px;">Scenario Probabilities</h3>
                <table style="width: 100%; font-size: 13px;">
                    {prob_html}
                </table>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                <h3 style="margin: 0 0 15px; color: #000; font-size: 16px; border-bottom: 2px solid #000; padding-bottom: 8px;">Greeks</h3>
                <table style="width: 100%; font-size: 13px;">
                    <tr style="background: #f5f5f5;">
                        <td style="padding: 8px;">
                            <b>Delta</b><br>
                            <span style="font-size: 10px; color: #666;">Sensitivity to Underlying Price</span>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">{greeks['delta']:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">
                            <b>Vega</b><br>
                            <span style="font-size: 10px; color: #666;">Sensitivity to Volatility (1%)</span>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">{greeks['vega']:.2f}</td>
                    </tr>
                    <tr style="background: #f5f5f5;">
                        <td style="padding: 8px;">
                            <b>Theta</b><br>
                            <span style="font-size: 10px; color: #666;">Time Decay (per day)</span>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">{greeks['theta']:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">
                            <b>Rho</b><br>
                            <span style="font-size: 10px; color: #666;">Sensitivity to Interest Rates</span>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">{greeks['rho']:.4f}</td>
                    </tr>
                </table>
            </div>

            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                <h3 style="margin: 0 0 15px; color: #c00; font-size: 16px; border-bottom: 2px solid #c00; padding-bottom: 8px;">Risk Metrics</h3>
                <table style="width: 100%; font-size: 13px;">
                    <tr style="background: #f5f5f5;">
                        <td style="padding: 8px;">
                            <b>VaR (95%)</b><br>
                            <span style="font-size: 10px; color: #666;">Worst loss in 95% of cases</span>
                        </td>
                        <td style="text-align: right; color: #c00; font-weight: 600; vertical-align: middle;">{params.currency} {risks['var_95']:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">
                            <b>VaR (99%)</b><br>
                            <span style="font-size: 10px; color: #666;">Worst loss in 99% of cases (Tail Risk)</span>
                        </td>
                        <td style="text-align: right; color: #c00; font-weight: 600; vertical-align: middle;">{params.currency} {risks['var_99']:,.2f}</td>
                    </tr>
                    <tr style="background: #f5f5f5;">
                        <td style="padding: 8px;">
                            <b>Expected Shortfall</b><br>
                            <span style="font-size: 10px; color: #666;">Avg loss when 95% VaR is breached</span>
                        </td>
                        <td style="text-align: right; color: #c00; font-weight: 600; vertical-align: middle;">{params.currency} {risks['expected_shortfall_95']:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">
                            <b>Max Loss</b><br>
                            <span style="font-size: 10px; color: #666;">Worst single outcome in {params.n_simulations:,} sims</span>
                        </td>
                        <td style="text-align: right; color: #c00; font-weight: 600; vertical-align: middle;">{params.currency} {risks['max_loss']:,.2f}</td>
                    </tr>
                </table>
            </div>
        </div>

        <div style="background: linear-gradient(135deg, {rec_color}15 0%, {rec_color}05 100%); border: 2px solid {rec_color}; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 10px; color: {rec_color}; font-size: 18px;">Recommendation: {recommendation}</h3>
            <div style="display: flex; gap: 30px; font-size: 13px;">
                <div><b>Investment:</b> {params.currency} {params.notional:,.2f}</div>
                <div><b>Fair Value:</b> {params.currency} {results['fair_value']:,.2f}</div>
                <div><b>Expected P&L:</b> <span style="color: {'#000' if expected_return >= 0 else '#c00'};">{params.currency} {results['fair_value'] - params.notional:+,.2f}</span></div>
            </div>
        </div>

        <div style="background: #fff8e1; border: 1px solid #ff9800; border-radius: 8px; padding: 12px;">
            <p style="margin: 0; font-size: 11px; color: #333;">
                <b>Disclaimer:</b> This analysis uses AI-determined pricing methodology and assumed market parameters. Consult a financial advisor.
            </p>
        </div>
    </div>'''

    return html

def generate_comprehensive_json(filename, key_df, dates_df, viz_data, extra_tags, params, results, greeks, risks):
    """Generate JSON with all data."""

    key_terms_list = []
    if key_df is not None and not key_df.empty:
        for _, row in key_df.iterrows():
            key_terms_list.append({"term": str(row.get('Term', '')), "value": str(row.get('Value', ''))})

    dates_list = []
    if dates_df is not None and not dates_df.empty:
        for _, row in dates_df.iterrows():
            dates_list.append({
                "date_type": str(row.get('Date Type', '')),
                "dates": str(row.get('Dates', row.get('Date', ''))),
            })

    return {
        "report_metadata": {
            "filename": filename,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "5.0"
        },
        "task1_extraction": {
            "key_terms": key_terms_list,
            "dates_schedule": dates_list,
            "observation_dates": params.observation_dates
        },
        "task2_analysis": {
            "product_classification": viz_data.get('product_classification', {}) if viz_data else {},
            "payoff_logic": viz_data.get('payoff_logic', {}) if viz_data else {}
        },
        "task3_pricing": {
            "pricing_methodology": params.pricing_methodology,
            "product_category": params.product_category,
            "parameters": {
                "notional": params.notional,
                "currency": params.currency,
                "maturity_years": params.maturity_years,
                "coupon_rate": params.coupon_rate,
                "coupon_type": params.coupon_type,
                "ko_barrier": params.ko_barrier,
                "ki_barrier": params.ki_barrier,
                "coupon_barrier": params.coupon_barrier,
                "protection_level": params.protection_level,
                "has_autocall": params.has_autocall,
                "has_knock_in": params.has_knock_in,
                "has_capital_protection": params.has_capital_protection,
            },
            "valuation": {
                "fair_value": results['fair_value'],
                "expected_return": results['expected_return'],
                "confidence_interval": list(results['confidence_interval_95']),
            },
            "probabilities": {
                "prob_autocall": results['prob_autocall'],
                "prob_knock_in": results['prob_ki'],
                "prob_loss": results['prob_loss'],
            },
            "greeks": greeks,
            "risk_metrics": risks,
        }
    }

def generate_comprehensive_pdf(filename, key_df, dates_df, viz_data, extra_tags, params, results, greeks, risks, scenario_images, flowchart_image):
    """Generate PDF report."""

    base_filename = os.path.splitext(filename)[0]
    pdf_filename = f"comprehensive_report_{base_filename}.pdf"

    try:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

        doc = SimpleDocTemplate(pdf_filename, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.6*inch, rightMargin=0.6*inch)
        styles = getSampleStyleSheet()
        story = []

        h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=16, textColor=colors.black, spaceBefore=15, spaceAfter=10)
        h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, textColor=colors.black, spaceBefore=10, spaceAfter=8)
        body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=12, alignment=TA_JUSTIFY, wordWrap='CJK')
        cell = ParagraphStyle('Cell', parent=styles['Normal'], fontSize=8, leading=10, wordWrap='CJK')
        header_cell = ParagraphStyle('HCell', parent=styles['Normal'], fontSize=8, fontName='Helvetica-Bold', textColor=colors.white)

        def black_header():
            return TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.black),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ])

        # Cover
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("STRUCTURED PRODUCT ANALYSIS", ParagraphStyle('Title', fontSize=22, alignment=TA_CENTER)))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"{params.product_category.replace('_', ' ').title()}", ParagraphStyle('Sub', fontSize=14, alignment=TA_CENTER, textColor=colors.gray)))
        story.append(Spacer(1, 0.3*inch))

        rec = "ATTRACTIVE" if results['expected_return'] > 0.02 else "NEUTRAL" if results['expected_return'] > -0.01 else "REVIEW"
        story.append(Paragraph(f"<b>RECOMMENDATION: {rec}</b>", ParagraphStyle('Rec', fontSize=16, alignment=TA_CENTER)))
        story.append(Paragraph(f"Fair Value: {params.currency} {results['fair_value']:,.2f} | Expected Return: {results['expected_return']:+.2%}", ParagraphStyle('Val', fontSize=11, alignment=TA_CENTER)))
        story.append(PageBreak())

        # Content (abbreviated for space)
        story.append(Paragraph("1. Parameter Extraction", h1))
        if key_df is not None and not key_df.empty:
            kt_data = [[Paragraph("<b>Term</b>", header_cell), Paragraph("<b>Value</b>", header_cell)]]
            for _, row in key_df.iterrows():
                kt_data.append([Paragraph(str(row.get('Term', '')), cell), Paragraph(str(row.get('Value', '')), cell)])
            kt_table = Table(kt_data, colWidths=[2.4*inch, 4.0*inch])
            kt_table.setStyle(black_header())
            story.append(kt_table)

        story.append(PageBreak())
        story.append(Paragraph("2. Monte Carlo Pricing", h1))
        story.append(Paragraph(f"Product Category: {params.product_category}", body))

        val_data = [
            [Paragraph("<b>Metric</b>", header_cell), Paragraph("<b>Value</b>", header_cell)],
            [Paragraph("Fair Value", cell), Paragraph(f"{params.currency} {results['fair_value']:,.2f}", cell)],
            [Paragraph("Expected Return", cell), Paragraph(f"{results['expected_return']:+.2%}", cell)],
        ]
        val_table = Table(val_data, colWidths=[2.5*inch, 2.5*inch])
        val_table.setStyle(black_header())
        story.append(val_table)

        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Disclaimer: This is for informational purposes only.", ParagraphStyle('Disc', fontSize=7, textColor=colors.gray)))

        doc.build(story)
        return pdf_filename

    except Exception as e:
        print(f"  PDF error: {e}")
        return None

def run_task3_pricing(key_df, dates_df, viz_data, extra_tags, filename, scenario_images=None, flowchart_image=None):
    """Run Task 3 with prompt engineering for any product type."""
    print("Running Task 3 (Monte Carlo Pricing with AI Methodology)...")

    # Extract parameters with LLM methodology
    extractor = PricingParameterExtractor(Config)
    params = extractor.extract_all(key_df, dates_df, viz_data, extra_tags)

    print(f"  Product Category: {params.product_category}")
    print(f"  Features: autocall={params.has_autocall}, KI={params.has_knock_in}, KO={params.has_knock_out}, protection={params.has_capital_protection}")
    print(f"  Coupon: {params.coupon_rate:.4%} ({params.coupon_type})")
    if params.ko_barrier:
        print(f"  KO Barrier: {params.ko_barrier:.2%}")
    if params.ki_barrier:
        print(f"  KI Barrier: {params.ki_barrier:.2%}")

    # Run pricing
    pricer = UniversalMonteCarloPricer(params)
    results = pricer.price()
    greeks = pricer.calculate_greeks()
    risks = pricer.calculate_risk_metrics()

    # Display
    display(Markdown("## 3. Monte Carlo Pricing Analysis (Task 3)"))
    html = generate_pricing_html_report(params, results, greeks, risks)
    display(HTML(html))

    # Generate outputs
    base_filename = os.path.splitext(filename)[0]
    json_filename = f"comprehensive_report_{base_filename}.json"
    json_data = generate_comprehensive_json(filename, key_df, dates_df, viz_data, extra_tags, params, results, greeks, risks)
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  JSON: {json_filename}")

    pdf_filename = generate_comprehensive_pdf(filename, key_df, dates_df, viz_data, extra_tags, params, results, greeks, risks, scenario_images, flowchart_image)
    if pdf_filename:
        print(f"  PDF: {pdf_filename}")

    display(Markdown("---"))
    display(Markdown(f"### Download: `{json_filename}` | `{pdf_filename}`"))

    try:
        from google.colab import files
        files.download(json_filename)
        if pdf_filename:
            files.download(pdf_filename)
    except:
        pass

    return {'params': params, 'results': results, 'greeks': greeks, 'risks': risks}

def generate_unified_report(filename, content):
    clear_output()

    # 记录总开始时间
    total_start = time.time()

    html_banner = f"""
    <div style="background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
                color: white; padding: 20px; border-radius: 10px; font-family: sans-serif; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="margin:0; font-size: 24px;">🚀 AI Financial Product Analyst</h2>
        <p style="margin:5px 0 0; font-size: 14px; opacity: 0.9;">
            <b>File:</b> {filename} | <b>Engine:</b> {Config.MODEL}
        </p>
    </div>
    """
    display(HTML(html_banner))

    StreamDisplay.write(f"📄 Processing File: {filename}...", speed=0.03)

    # 计时 1: 读取 PDF
    with BlockTimer("PDF Parsing"):
        text = read_pdf_text(content)
        if not text.strip():
            StreamDisplay.write("❌ Error: Could not extract text from PDF.", color="red")
            return
        issuer_fallback = fallback_extract_issuer(text)

    # =========================================================
    # PART 1: TASK 1
    # =========================================================
    StreamDisplay.write("⚡ Running Task 1: Neural Parameter Extraction...", speed=0.02)
    client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)

    key_df, dates_df = pd.DataFrame(), pd.DataFrame()

    # 计时 2: Task 1 LLM 调用 (通常是网络/API瓶颈)
    with BlockTimer("Task 1 (LLM Extraction)"):
        try:
            raw_task1 = call_task1_llm(client, text)
            key_df = markdown_table_to_df(extract_table_block(raw_task1, "KEY_TERMS_TABLE"))
            dates_df = markdown_table_to_df(extract_table_block(raw_task1, "DATES_TABLE"))
            key_df = ensure_issuer_row(key_df, issuer_fallback)
            key_df = merge_synonyms_same_value(key_df)
            key_df = filter_key_terms_core(key_df)
            dates_df = enforce_observation_dates(dates_df, text)
            dates_df = compress_dates_table(dates_df)
            dates_df = filter_dates_for_summary(dates_df)
        except Exception as e:
            print(f"Task 1 Error: {e}")

    # 输出 Task 1 结果 (不计时，因为是 UI 渲染)
    display(Markdown("---"))
    StreamDisplay.write(f"# Derivative Product Analysis: {filename}", speed=0.05)
    StreamDisplay.write("## 1. Key Quantified Parameters (Task 1 Output)", speed=0.03)

    if not key_df.empty:
        render_task1_table(key_df[["Term", "Value"]], caption="Product Information")
    else:
        display(Markdown("_No key terms extracted._"))

    if not dates_df.empty:
        cols = [c for c in dates_df.columns if c]
        render_task1_table(dates_df[cols], caption="Key Dates")
    else:
        display(Markdown("_No explicit dates extracted._"))

    display(Markdown("---"))

    # =========================================================
    # PART 2: TASK 2
    # =========================================================
    extra_tags = extract_structural_tags(text)
    StreamDisplay.write("🧠 Running Task 2: Structural Logic & Visual Analysis...", speed=0.02)
    analyst = LogicArchitect(Config)

    viz_data = None
    scenario_image_paths = []
    flowchart_image_path = None

    # 计时 3: Task 2 LLM 分析 (通常是 API 生成长文本瓶颈)
    with BlockTimer("Task 2 (Logic Analysis API)"):
        viz_data = analyst.analyze_structure(text, key_df, dates_df, extra_tags)

    if viz_data:
        StreamDisplay.write("## 2. Structural & Payoff Analysis (Task 2 Output)", speed=0.03)

        prod_class = viz_data.get("product_classification", {})
        payoff = viz_data.get('payoff_logic', {})

        if prod_class:
            StreamDisplay.write("### 2.1 Product Analysis", speed=0.02)
            analysis_text = (
                f"**Product Type:** {prod_class.get('type')}\n"
                f"**Confidence:** {prod_class.get('confidence')}\n"
                f"**Reasoning:** {prod_class.get('reasoning')}"
            )
            StreamDisplay.write(analysis_text, speed=0.01)

        display(Markdown("---"))

        # 计时 4: 流程图生成 (Graphviz 渲染瓶颈)
        with BlockTimer("Flowchart Rendering"):
            flow_viz = LifecycleFlowVisualizer()
            flow_img_path = flow_viz.plot_logic_flow(viz_data)
            flowchart_image_path = flow_img_path

        StreamDisplay.write("### 2.2 Lifecycle & Logic Flow", speed=0.02)
        if flow_img_path and os.path.exists(flow_img_path):
            display(Image(flow_img_path))
        else:
            display(Markdown("_Flowchart could not be generated._"))

        # 4. Formulas
        formulas = payoff.get('formulas', [])
        if formulas:
            display(Markdown("### 2.3 Payoff Formulas & Logic"))

            calendar = viz_data.get("calendar", []) or []
            early_redemption_dates = []

            for entry in calendar:
                date_str = entry.get("date")
                if not date_str:
                    continue

                levels = entry.get("levels", {}) or {}
                event_str = entry.get("event", "") or ""

                levels_keys = " ".join(levels.keys()).lower()

                if (
                    "knock-out" in levels_keys
                    or "autocall" in levels_keys
                    or "early redemption" in event_str.lower()
                ):
                    early_redemption_dates.append(date_str)

            early_redemption_dates = sorted(set(early_redemption_dates))

            EARLY_REDEMPTION_KEYWORDS = [
                "early redemption",
                "automatic early redemption",
                "auto early redemption",
                "auto-redemption",
                "auto redemption",

                "autocall",
                "auto-call",
                "auto call",
                "early autocall",
                "autocall payoff",
                "autocall amount",

                "early call",

                "knock-out redemption",
                "knock out redemption",
                "ko redemption",
            ]

            inserted_er_table = False

            for f in formulas:
              label = f.get("label", "")
              label_lower = label.lower()

              display(Markdown(f"**{label}**"))
              display(Markdown(f"_{f.get('narrative')}_"))

              if (
                early_redemption_dates
                and not inserted_er_table
                and any(kw in label_lower for kw in EARLY_REDEMPTION_KEYWORDS)
              ):
                display(Markdown(
                  "**Settlement Dates (Early Redemption – read left to right, earliest → latest)**"
                ))

                cols_per_row = 4

                rows = []
                for i in range(0, len(early_redemption_dates), cols_per_row):
                  chunk = early_redemption_dates[i:i + cols_per_row]
                  while len(chunk) < cols_per_row:
                    chunk.append("")
                  rows.append(chunk)

                header_row = "| " + " | ".join([""] * cols_per_row) + " |"
                separator_row = "| " + " | ".join(["---"] * cols_per_row) + " |"

                md_lines = [header_row, separator_row]
                for r in rows:
                  md_lines.append("| " + " | ".join(r) + " |")

                display(Markdown("\n".join(md_lines)))
                inserted_er_table = True

              display(Markdown("---"))


        # 5. Scenarios
        scenarios = payoff.get('mece_scenarios', [])
        if scenarios:
            display(Markdown(f"### 2.4 Visual Simulation of {len(scenarios)} MECE Market Outcomes"))
            display(Markdown(
                "_All scenarios show the complete observation schedule (x-axis synchronized). "
                "Y-axis standardized to 0-110%. "
                "Orange vertical line indicates product termination; grayed dates indicate post-termination observations._"
            ))
            display(Markdown("---"))

            viz = EducationalVisualizer()
            for i, scen in enumerate(scenarios, 1):
                try:
                    img_path = viz.plot_professional_scenario(scen, viz_data, index=i)
                    display(Markdown(f"#### Scenario {i}: {scen.get('condition')}"))
                    display(Markdown(f"**Expected Outcome:** {scen.get('outcome', 'N/A')}"))
                    if img_path and os.path.exists(img_path):
                        display(Image(img_path))
                    else:
                        display(Markdown("_Chart generation failed._"))
                except Exception as e:
                    display(Markdown(f"_Error generating scenario {i}: {str(e)}_"))
                display(Markdown("---"))


        # TASK 3: PRICING
        # 计时 6: 蒙特卡洛模拟 (CPU 计算瓶颈)
        with BlockTimer("Task 3 (Monte Carlo Simulation)"):
            try:
                run_task3_pricing(
                    key_df, dates_df, viz_data, extra_tags, filename,
                    scenario_images=scenario_image_paths,
                    flowchart_image=flowchart_image_path
                )
            except Exception as e:
                StreamDisplay.write(f"❌ Pricing Error: {str(e)}", color="red")
                traceback.print_exc()

    else:
        StreamDisplay.write("❌ Task 2 Analysis failed to generate structured data.", color="red")

    # 打印总耗时
    total_elapsed = time.time() - total_start
    StreamDisplay.write(f"🏁 All Tasks Completed in {total_elapsed:.2f}s", speed=0.05, color="#2196F3")
