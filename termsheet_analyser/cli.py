"""Terminal entry points for offline inspection and the actual PDF workflow."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if len(argv)==1 and not argv[0].startswith("-") and argv[0] not in {"tags", "analyse"}:
        argv.insert(0, "tags")  # preserve the original one-file offline command
    parser=argparse.ArgumentParser(description="Derivative term-sheet research workflow")
    commands=parser.add_subparsers(dest="command", required=True)
    tags=commands.add_parser("tags", help="Extract structural regex hints from text without an API")
    tags.add_argument("text_file", type=Path)
    analyse=commands.add_parser("analyse", help="Run the PDF extraction and structural-analysis workflow")
    analyse.add_argument("pdf", type=Path)
    analyse.add_argument("--output", default="outputs", type=Path)
    analyse.add_argument("--no-charts", action="store_true")
    analyse.add_argument("--include-pricing", action="store_true", help="Enable the unvalidated research pricer; read docs/CODE_REVIEW.md first")
    args=parser.parse_args(argv)
    if args.command=="tags":
        from .step02_extraction import extract_structural_tags
        print(json.dumps(extract_structural_tags(args.text_file.read_text(encoding="utf-8")), indent=2))
        return 0
    from .workflow import run_workflow
    try:
        result=run_workflow(args.pdf,args.output,render_charts=not args.no_charts,include_pricing=args.include_pricing)
    except Exception as exc:
        parser.exit(1, f"Workflow stopped: {exc}\n")
    print(f"Analysis saved: {result.analysis_file}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    return 0
