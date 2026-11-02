"""Compatibility command; use python -m termsheet_analyser for the new interface."""
from termsheet_analyser.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
