"""02 — Extract term tables and normalise terms and observation dates."""
from __future__ import annotations

from .config import Config as _Config
Config = _Config()
from .prompts import TASK1_SYSTEM_PROMPT
from .prompts import TASK1_USER_PROMPT
from typing import Dict
import pandas as pd
import re

def extract_structural_tags(text: str) -> dict[str, bool | str]:
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

def call_task1_llm(client, text: str, model: str | None = None) -> str:
    resp = client.chat.completions.create(
        model=model or Config.MODEL,
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
