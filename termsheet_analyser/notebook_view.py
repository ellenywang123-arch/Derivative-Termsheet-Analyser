"""Notebook presentation adapter for the original combined workflow."""
from __future__ import annotations

from .config import Config as _Config
Config = _Config()
from .presentation import BlockTimer
from .presentation import HTML
from .presentation import Image
from .presentation import Markdown
from .presentation import StreamDisplay
from .presentation import clear_output
from .presentation import display
from .step01_pdf import fallback_extract_issuer
from .step01_pdf import read_pdf_text
from .step02_extraction import call_task1_llm
from .step02_extraction import compress_dates_table
from .step02_extraction import enforce_observation_dates
from .step02_extraction import ensure_issuer_row
from .step02_extraction import extract_structural_tags
from .step02_extraction import extract_table_block
from .step02_extraction import filter_dates_for_summary
from .step02_extraction import filter_key_terms_core
from .step02_extraction import markdown_table_to_df
from .step02_extraction import merge_synonyms_same_value
from .step03_structure import LogicArchitect
from .step05_visualization import EducationalVisualizer
from .step05_visualization import LifecycleFlowVisualizer
from .step05_visualization import render_task1_table
from .step08_reporting import run_task3_pricing
from openai import OpenAI
import os
import pandas as pd
import time
import traceback

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
            raise RuntimeError("Task 1 extraction failed; workflow stopped.") from e

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
            display(Image(filename=flow_img_path))
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
                        scenario_image_paths.append(img_path)
                        display(Image(filename=img_path))
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
