"""Selected original functions from the Team C notebook."""
import re
from typing import Dict

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

def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_value(v: str) -> str:
    v = (v or "").strip().lower()
    v = v.replace(" ", "")
    v = v.replace(",", "")
    return v
