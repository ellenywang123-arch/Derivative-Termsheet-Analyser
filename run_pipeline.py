"""Experimental PDF workflow; requires dependencies, Graphviz and API configuration."""
import argparse, os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output",type=Path,default=Path("outputs"))
    args=parser.parse_args()
    if not os.environ.get("TERMSHEET_API_KEY"):
        parser.error("Set TERMSHEET_API_KEY in your environment first.")
    if args.pdf.suffix.lower() != ".pdf" or not args.pdf.is_file():
        parser.error("Provide an existing PDF file.")
    import research_pipeline as pipeline
    args.output.mkdir(exist_ok=True,parents=True)
    counter=0
    def capture(*objects, **kwargs):
        nonlocal counter
        for obj in objects:
            counter+=1
            # HTML display objects expose data; other notebook objects are text.
            data=getattr(obj,"data",None)
            suffix=".html" if isinstance(data,str) and hasattr(obj,"_repr_html_") else ".txt"
            (args.output/f"display-{counter:03d}{suffix}").write_text(data if isinstance(data,str) else str(obj),encoding="utf-8")
    pipeline.display=capture
    pipeline.generate_unified_report(args.pdf.name,args.pdf.read_bytes())
    print("Workflow returned. Inspect generated outputs and logs; this does not certify model accuracy.")

if __name__ == "__main__":
    main()
