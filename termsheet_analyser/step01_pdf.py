"""01 — Read PDF text and identify the issuer."""
from __future__ import annotations

from .config import Config as _Config
Config = _Config()
import fitz
import re

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
