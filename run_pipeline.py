"""Compatibility PDF command: python run_pipeline.py document.pdf --output outputs"""
import sys
from termsheet_analyser.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["analyse", *sys.argv[1:]]))
