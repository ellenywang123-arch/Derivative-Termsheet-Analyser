"""Read this file first: the ordered Python research workflow.

PDF → extraction → structure → illustrations → optional pricing → saved artifacts.
No website or browser demo is involved. API calls only occur when run_workflow is called.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from openai import OpenAI

from .config import Config
from .step01_pdf import fallback_extract_issuer, read_pdf_text
from .step02_extraction import (
    call_task1_llm, ensure_issuer_row, extract_structural_tags, extract_table_block,
    markdown_table_to_df, merge_synonyms_same_value,
)
from .step03_structure import LogicArchitect


@dataclass
class WorkflowResult:
    output_directory: Path
    analysis_file: Path
    warnings: list[str]
    pricing: dict | None = None


def run_workflow(pdf_path, output_directory="outputs", *, config=None,
                 client=None, render_charts=True, include_pricing=False):
    """Run with an authorised PDF and environment credentials.

    Pricing is opt-in because the preserved research pricer has unresolved economic
    assumptions documented in docs/CODE_REVIEW.md. Failures stop the computation;
    they never silently substitute a different product.
    """
    settings = config or Config()
    settings.validate()
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Provide an existing PDF file.")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    warnings = []

    # 01. PDF → text. Never silently ignore the end of a long document.
    text = read_pdf_text(str(pdf_path), max_chars=settings.MAX_CHARS + 1)
    if not text.strip():
        raise ValueError("No PDF text found. Scanned PDFs require OCR first.")
    if len(text) > settings.MAX_CHARS:
        raise ValueError("PDF exceeds the configured character limit; split it explicitly before analysis.")

    # 02. LLM → two tables. Save original extraction before normalisation.
    client = client if client is not None else OpenAI(api_key=settings.API_KEY, base_url=settings.BASE_URL)
    raw = call_task1_llm(client, text, model=settings.MODEL)
    key_terms = markdown_table_to_df(extract_table_block(raw, "KEY_TERMS_TABLE"))
    dates = markdown_table_to_df(extract_table_block(raw, "DATES_TABLE"))
    required_terms = {"Term", "Value"}
    if key_terms.empty or not required_terms.issubset(key_terms.columns):
        raise ValueError("Extraction did not produce a non-empty Term/Value table.")
    if not dates.empty and not {"Date Type", "Date"}.issubset(dates.columns):
        raise ValueError("Date table must have Date Type and Date columns.")
    (output / "02_raw_extraction.txt").write_text(raw, encoding="utf-8")
    key_terms = ensure_issuer_row(key_terms, fallback_extract_issuer(text))
    key_terms = merge_synonyms_same_value(key_terms)
    key_terms.to_csv(output / "02_key_terms.csv", index=False)
    # Preserve row-level date/event/barrier associations. Summary compression and
    # inferred date expansion are intentionally excluded from this workflow.
    dates.to_csv(output / "02_dates.csv", index=False)
    if dates.empty:
        warnings.append("No dated schedule extracted; event timing is unverified.")

    # 03. Tables + source text → product structure and payoff explanation.
    tags = extract_structural_tags(text)
    structure = LogicArchitect(settings, client=client).analyze_structure(text, key_terms, dates, tags)
    if not isinstance(structure, dict) or not isinstance(structure.get("product_classification"), dict):
        raise ValueError("Structural analysis did not produce a product classification object.")

    # 04–05. Illustrative paths and charts, saved under this run's output directory.
    scenario_images = []
    flowchart = None
    if render_charts:
        from .step05_visualization import EducationalVisualizer, LifecycleFlowVisualizer
        flowchart = LifecycleFlowVisualizer().plot_logic_flow(structure, str(output / "05_lifecycle"))
        if flowchart is None:
            warnings.append("Lifecycle chart unavailable; check Graphviz and the generated flowchart specification.")
        visualizer = EducationalVisualizer()
        for index, scenario in enumerate(structure.get("payoff_logic", {}).get("mece_scenarios", []), 1):
            image = visualizer.plot_professional_scenario(scenario, structure, index=index, output_directory=output)
            if image:
                scenario_images.append(str(image))

    # 06–08. Optional research pricing, plus machine-readable report artifacts.
    pricing = None
    if include_pricing:
        from .step08_reporting import run_task3_pricing
        warnings.append("Experimental pricing enabled. Greeks, event timing, cashflow conventions and risk measures are not validated.")
        pricing = run_task3_pricing(key_terms, dates, structure, tags, pdf_path.name,
                                   scenario_images=scenario_images, flowchart_image=flowchart,
                                   config=settings, output_directory=output)
    payload = {
        "source_filename": pdf_path.name,
        "structural_tags": tags,
        "key_terms": key_terms.to_dict(orient="records"),
        "dated_events": dates.to_dict(orient="records"),
        "structure": structure,
        "scenario_images": scenario_images,
        "flowchart": flowchart,
        "pricing_enabled": include_pricing,
        "warnings": warnings,
    }
    analysis_file = output / "analysis.json"
    analysis_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return WorkflowResult(output, analysis_file, warnings, pricing)
