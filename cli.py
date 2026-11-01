"""Offline structural tag demo: python cli.py examples/synthetic_termsheet.txt"""
import argparse, json
from pathlib import Path
from termsheet_core import extract_structural_tags

def main():
    parser=argparse.ArgumentParser(description="Offline structural tag extraction (no LLM or PDF parsing).")
    parser.add_argument("text_file",type=Path)
    args=parser.parse_args()
    print(json.dumps(extract_structural_tags(args.text_file.read_text(encoding="utf-8")),indent=2))

if __name__ == "__main__":
    main()
