"""08 — Export pricing HTML/JSON/PDF research reports."""
from __future__ import annotations

from .config import Config as _Config
Config = _Config()
from .step06_parameters import PricingParameterExtractor
from .step07_pricing import UniversalMonteCarloPricer
from datetime import datetime
import json
import os

def generate_pricing_html_report(params, results, greeks, risks):
    """Generate HTML report adapted to product type with Risk & Greek Explanations."""

    expected_return = results.get('expected_return', 0)
    if expected_return > 0.02:
        recommendation, rec_color = "UNVALIDATED RESEARCH OUTPUT", "#dc3545"
    elif expected_return > -0.01:
        recommendation, rec_color = "UNVALIDATED RESEARCH OUTPUT", "#dc3545"
    else:
        recommendation, rec_color = "UNVALIDATED RESEARCH OUTPUT", "#dc3545"

    # Product description
    category = params.product_category
    features = []
    if params.is_worst_of:
        features.append("Worst-of")
    if params.has_memory_feature:
        features.append("Memory")
    if params.has_capital_protection:
        features.append("Capital Protected")
    if params.has_participation:
        features.append(f"Participation ({params.participation_rate:.0%})")
    if params.has_autocall:
        features.append("Autocallable")

    product_desc = f"{' '.join(features)} {category.replace('_', ' ').title()}"

    # Methodology from LLM
    methodology = params.pricing_methodology
    payoff_desc = methodology.get('payoff_at_maturity', {}).get('description', 'N/A') if methodology else 'N/A'
    loss_condition = methodology.get('payoff_at_maturity', {}).get('loss_condition', 'N/A') if methodology else 'N/A'

    # Build barriers section (only if applicable)
    barriers_html = ""
    if params.has_knock_out or params.has_knock_in or params.has_coupon_barrier:
        barriers_html = f"""
        <tr>
            <td style="padding: 8px 0; color: #333;">KO Barrier</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.ko_barrier:.2%}" if params.ko_barrier else "N/A"}</td>
            <td style="padding: 8px 0 8px 30px; color: #333;">KI Barrier</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.ki_barrier:.2%}" if params.ki_barrier else "N/A"}</td>
        </tr>
        <tr>
            <td style="padding: 8px 0; color: #333;">Coupon Barrier</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.coupon_barrier:.2%}" if params.coupon_barrier else "N/A"}</td>
            <td style="padding: 8px 0 8px 30px; color: #333;">Protection Level</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.protection_level:.2%}" if params.protection_level else "N/A"}</td>
        </tr>
        """
    elif params.has_capital_protection:
        barriers_html = f"""
        <tr>
            <td style="padding: 8px 0; color: #333;">Protection Level</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.protection_level:.2%}" if params.protection_level else "100%"}</td>
            <td style="padding: 8px 0 8px 30px; color: #333;">Cap Level</td>
            <td style="padding: 8px 0; text-align: right; font-weight: 600;">{f"{params.cap_level:.2%}" if params.cap_level else "N/A"}</td>
        </tr>
        """

    # Probabilities section (only show relevant ones)
    prob_html = ""
    if params.has_autocall:
        prob_html += f'<tr><td style="padding: 10px 0;">Prob. Autocall</td><td style="text-align: right;"><span style="background: #e0e0e0; padding: 3px 10px; border-radius: 10px; font-weight: 600;">{results["prob_autocall"]:.1%}</span></td></tr>'
    if params.has_knock_in:
        prob_html += f'<tr><td style="padding: 10px 0;">Prob. Knock-In</td><td style="text-align: right;"><span style="background: #ffe0b2; padding: 3px 10px; border-radius: 10px; font-weight: 600;">{results["prob_ki"]:.1%}</span></td></tr>'
    if results['prob_loss'] > 0 or not params.has_capital_protection:
        prob_html += f'<tr><td style="padding: 10px 0;">Prob. Loss</td><td style="text-align: right;"><span style="background: #ffcdd2; color: #c00; padding: 3px 10px; border-radius: 10px; font-weight: 600;">{results["prob_loss"]:.1%}</span></td></tr>'

    prob_safe = max(0, 1 - results['prob_autocall'] - results['prob_loss'])
    prob_html += f'<tr><td style="padding: 10px 0;">Prob. Full Principal</td><td style="text-align: right;"><span style="background: #c8e6c9; padding: 3px 10px; border-radius: 10px; font-weight: 600;">{prob_safe:.1%}</span></td></tr>'

    html = f'''
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 950px; margin: 20px auto;">
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #333 100%); color: white; padding: 25px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0; font-size: 24px;">Monte Carlo Pricing Analysis (Task 3)</h2>
            <p style="margin: 8px 0 0; opacity: 0.9; font-size: 14px;">{product_desc}</p>
            <p style="margin: 5px 0 0; opacity: 0.8; font-size: 12px;">
                {params.n_simulations:,} Simulations | {params.n_underlyings} Underlying(s) | {params.maturity_years:.1f} Years
            </p>
        </div>

        <div style="background: #e3f2fd; border: 1px solid #1976d2; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px; color: #1565c0; font-size: 14px;">🤖 AI-Determined Pricing Methodology</h4>
            <p style="margin: 0 0 8px; font-size: 12px;"><b>Product Category:</b> {category}</p>
            <p style="margin: 0 0 8px; font-size: 12px;"><b>Payoff at Maturity:</b> {payoff_desc}</p>
            <p style="margin: 0; font-size: 12px;"><b>Loss Condition:</b> {loss_condition}</p>
        </div>

        <div style="background: #e8f5e9; border: 1px solid #4caf50; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px; color: #2e7d32; font-size: 14px;">✓ Parameters from Task 1</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td style="padding: 4px;"><b>Coupon Rate:</b></td>
                    <td>{params.coupon_rate:.4%} p.a. ({params.coupon_type})</td>
                </tr>
                {f'<tr><td style="padding: 4px;"><b>KO Barrier:</b></td><td>{params.ko_barrier:.2%}</td></tr>' if params.ko_barrier else ''}
                {f'<tr><td style="padding: 4px;"><b>KI Barrier:</b></td><td>{params.ki_barrier:.2%}</td></tr>' if params.ki_barrier else ''}
            </table>
        </div>

        <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 15px; color: #000; font-size: 16px; border-bottom: 2px solid #000; padding-bottom: 8px;">Product Configuration</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <tr>
                    <td style="padding: 8px 0; color: #333; width: 25%;">Notional</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600; width: 25%;">{params.currency} {params.notional:,.2f}</td>
                    <td style="padding: 8px 0 8px 30px; color: #333; width: 25%;">Maturity</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600; width: 25%;">{params.maturity_years:.2f} years</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #333;">Coupon Rate (p.a.)</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600;">{params.coupon_rate:.4%}</td>
                    <td style="padding: 8px 0 8px 30px; color: #333;">Coupon Type</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600;">{params.coupon_type}</td>
                </tr>
                {barriers_html}
                <tr>
                    <td style="padding: 8px 0; color: #333;">Volatility</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600;">{params.volatilities[0]:.1%}</td>
                    <td style="padding: 8px 0 8px 30px; color: #333;">Risk-Free Rate</td>
                    <td style="padding: 8px 0; text-align: right; font-weight: 600;">{params.risk_free_rate:.2%}</td>
                </tr>
            </table>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                <h3 style="margin: 0 0 15px; color: #000; font-size: 16px; border-bottom: 2px solid #000; padding-bottom: 8px;">Valuation Results</h3>
                <div style="text-align: center; padding: 15px 0; background: #f5f5f5; border-radius: 8px; margin-bottom: 15px;">
                    <div style="font-size: 32px; font-weight: 700; color: #000;">{params.currency} {results['fair_value']:,.2f}</div>
                    <div style="font-size: 12px; color: #333;">Fair Value (Per Unit)</div>
                </div>
                <table style="width: 100%; font-size: 12px;">
                    <tr><td style="padding: 6px 0;">95% CI</td><td style="text-align: right;">{params.currency} {results['confidence_interval_95'][0]:,.2f} - {results['confidence_interval_95'][1]:,.2f}</td></tr>
                    <tr style="background: #f5f5f5;"><td style="padding: 8px 0; font-weight: 600;">Expected Return</td><td style="text-align: right; font-weight: 700; font-size: 14px; color: {'#000' if expected_return >= 0 else '#c00'};">{expected_return:+.2%}</td></tr>
                </table>
            </div>

            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                <h3 style="margin: 0 0 15px; color: #000; font-size: 16px; border-bottom: 2px solid #000; padding-bottom: 8px;">Scenario Probabilities</h3>
                <table style="width: 100%; font-size: 13px;">
                    {prob_html}
                </table>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                <h3 style="margin: 0 0 15px; color: #000; font-size: 16px; border-bottom: 2px solid #000; padding-bottom: 8px;">Greeks</h3>
                <table style="width: 100%; font-size: 13px;">
                    <tr style="background: #f5f5f5;">
                        <td style="padding: 8px;">
                            <b>Delta</b><br>
                            <span style="font-size: 10px; color: #666;">Sensitivity to Underlying Price</span>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">{greeks['delta']:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">
                            <b>Vega</b><br>
                            <span style="font-size: 10px; color: #666;">Sensitivity to Volatility (1%)</span>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">{greeks['vega']:.2f}</td>
                    </tr>
                    <tr style="background: #f5f5f5;">
                        <td style="padding: 8px;">
                            <b>Theta</b><br>
                            <span style="font-size: 10px; color: #666;">Time Decay (per day)</span>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">{greeks['theta']:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">
                            <b>Rho</b><br>
                            <span style="font-size: 10px; color: #666;">Sensitivity to Interest Rates</span>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">{greeks['rho']:.4f}</td>
                    </tr>
                </table>
            </div>

            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                <h3 style="margin: 0 0 15px; color: #c00; font-size: 16px; border-bottom: 2px solid #c00; padding-bottom: 8px;">Risk Metrics</h3>
                <table style="width: 100%; font-size: 13px;">
                    <tr style="background: #f5f5f5;">
                        <td style="padding: 8px;">
                            <b>VaR (95%)</b><br>
                            <span style="font-size: 10px; color: #666;">Worst loss in 95% of cases</span>
                        </td>
                        <td style="text-align: right; color: #c00; font-weight: 600; vertical-align: middle;">{params.currency} {risks['var_95']:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">
                            <b>VaR (99%)</b><br>
                            <span style="font-size: 10px; color: #666;">Worst loss in 99% of cases (Tail Risk)</span>
                        </td>
                        <td style="text-align: right; color: #c00; font-weight: 600; vertical-align: middle;">{params.currency} {risks['var_99']:,.2f}</td>
                    </tr>
                    <tr style="background: #f5f5f5;">
                        <td style="padding: 8px;">
                            <b>Expected Shortfall</b><br>
                            <span style="font-size: 10px; color: #666;">Avg loss when 95% VaR is breached</span>
                        </td>
                        <td style="text-align: right; color: #c00; font-weight: 600; vertical-align: middle;">{params.currency} {risks['expected_shortfall_95']:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">
                            <b>Max Loss</b><br>
                            <span style="font-size: 10px; color: #666;">Worst single outcome in {params.n_simulations:,} sims</span>
                        </td>
                        <td style="text-align: right; color: #c00; font-weight: 600; vertical-align: middle;">{params.currency} {risks['max_loss']:,.2f}</td>
                    </tr>
                </table>
            </div>
        </div>

        <div style="background: linear-gradient(135deg, {rec_color}15 0%, {rec_color}05 100%); border: 2px solid {rec_color}; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 10px; color: {rec_color}; font-size: 18px;">Recommendation: {recommendation}</h3>
            <div style="display: flex; gap: 30px; font-size: 13px;">
                <div><b>Investment:</b> {params.currency} {params.notional:,.2f}</div>
                <div><b>Fair Value:</b> {params.currency} {results['fair_value']:,.2f}</div>
                <div><b>Expected P&L:</b> <span style="color: {'#000' if expected_return >= 0 else '#c00'};">{params.currency} {results['fair_value'] - params.notional:+,.2f}</span></div>
            </div>
        </div>

        <div style="background: #fff8e1; border: 1px solid #ff9800; border-radius: 8px; padding: 12px;">
            <p style="margin: 0; font-size: 11px; color: #333;">
                <b>Disclaimer:</b> This analysis uses AI-determined pricing methodology and assumed market parameters. Consult a financial advisor.
            </p>
        </div>
    </div>'''

    return html

def generate_comprehensive_json(filename, key_df, dates_df, viz_data, extra_tags, params, results, greeks, risks):
    """Generate JSON with all data."""

    key_terms_list = []
    if key_df is not None and not key_df.empty:
        for _, row in key_df.iterrows():
            key_terms_list.append({"term": str(row.get('Term', '')), "value": str(row.get('Value', ''))})

    dates_list = []
    if dates_df is not None and not dates_df.empty:
        for _, row in dates_df.iterrows():
            dates_list.append({
                "date_type": str(row.get('Date Type', '')),
                "dates": str(row.get('Dates', row.get('Date', ''))),
            })

    return {
        "report_metadata": {
            "filename": filename,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "5.0"
        },
        "task1_extraction": {
            "key_terms": key_terms_list,
            "dates_schedule": dates_list,
            "observation_dates": params.observation_dates
        },
        "task2_analysis": {
            "product_classification": viz_data.get('product_classification', {}) if viz_data else {},
            "payoff_logic": viz_data.get('payoff_logic', {}) if viz_data else {}
        },
        "task3_pricing": {
            "pricing_methodology": params.pricing_methodology,
            "product_category": params.product_category,
            "parameters": {
                "notional": params.notional,
                "currency": params.currency,
                "maturity_years": params.maturity_years,
                "coupon_rate": params.coupon_rate,
                "coupon_type": params.coupon_type,
                "ko_barrier": params.ko_barrier,
                "ki_barrier": params.ki_barrier,
                "coupon_barrier": params.coupon_barrier,
                "protection_level": params.protection_level,
                "has_autocall": params.has_autocall,
                "has_knock_in": params.has_knock_in,
                "has_capital_protection": params.has_capital_protection,
            },
            "valuation": {
                "fair_value": results['fair_value'],
                "expected_return": results['expected_return'],
                "confidence_interval": list(results['confidence_interval_95']),
            },
            "probabilities": {
                "prob_autocall": results['prob_autocall'],
                "prob_knock_in": results['prob_ki'],
                "prob_loss": results['prob_loss'],
            },
            "greeks": greeks,
            "risk_metrics": risks,
        }
    }

def generate_comprehensive_pdf(filename, key_df, dates_df, viz_data, extra_tags, params, results, greeks, risks, scenario_images, flowchart_image, output_directory=None):
    """Generate PDF report."""

    base_filename = os.path.splitext(filename)[0]
    pdf_filename = os.path.join(str(output_directory or "."), f"comprehensive_report_{base_filename}.pdf")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

        doc = SimpleDocTemplate(pdf_filename, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.6*inch, rightMargin=0.6*inch)
        styles = getSampleStyleSheet()
        story = []

        h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=16, textColor=colors.black, spaceBefore=15, spaceAfter=10)
        h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, textColor=colors.black, spaceBefore=10, spaceAfter=8)
        body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=12, alignment=TA_JUSTIFY, wordWrap='CJK')
        cell = ParagraphStyle('Cell', parent=styles['Normal'], fontSize=8, leading=10, wordWrap='CJK')
        header_cell = ParagraphStyle('HCell', parent=styles['Normal'], fontSize=8, fontName='Helvetica-Bold', textColor=colors.white)

        def black_header():
            return TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.black),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ])

        # Cover
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("STRUCTURED PRODUCT ANALYSIS", ParagraphStyle('Title', fontSize=22, alignment=TA_CENTER)))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"{params.product_category.replace('_', ' ').title()}", ParagraphStyle('Sub', fontSize=14, alignment=TA_CENTER, textColor=colors.gray)))
        story.append(Spacer(1, 0.3*inch))

        rec = "UNVALIDATED RESEARCH OUTPUT"
        story.append(Paragraph(f"<b>STATUS: {rec}</b>", ParagraphStyle('Rec', fontSize=16, alignment=TA_CENTER)))
        story.append(Paragraph(f"Fair Value: {params.currency} {results['fair_value']:,.2f} | Expected Return: {results['expected_return']:+.2%}", ParagraphStyle('Val', fontSize=11, alignment=TA_CENTER)))
        story.append(PageBreak())

        # Content (abbreviated for space)
        story.append(Paragraph("1. Parameter Extraction", h1))
        if key_df is not None and not key_df.empty:
            kt_data = [[Paragraph("<b>Term</b>", header_cell), Paragraph("<b>Value</b>", header_cell)]]
            for _, row in key_df.iterrows():
                kt_data.append([Paragraph(str(row.get('Term', '')), cell), Paragraph(str(row.get('Value', '')), cell)])
            kt_table = Table(kt_data, colWidths=[2.4*inch, 4.0*inch])
            kt_table.setStyle(black_header())
            story.append(kt_table)

        story.append(PageBreak())
        story.append(Paragraph("2. Monte Carlo Pricing", h1))
        story.append(Paragraph(f"Product Category: {params.product_category}", body))

        val_data = [
            [Paragraph("<b>Metric</b>", header_cell), Paragraph("<b>Value</b>", header_cell)],
            [Paragraph("Fair Value", cell), Paragraph(f"{params.currency} {results['fair_value']:,.2f}", cell)],
            [Paragraph("Expected Return", cell), Paragraph(f"{results['expected_return']:+.2%}", cell)],
        ]
        val_table = Table(val_data, colWidths=[2.5*inch, 2.5*inch])
        val_table.setStyle(black_header())
        story.append(val_table)

        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Disclaimer: This is for informational purposes only.", ParagraphStyle('Disc', fontSize=7, textColor=colors.gray)))

        doc.build(story)
        return pdf_filename

    except Exception as e:
        print(f"  PDF error: {e}")
        return None

def run_task3_pricing(key_df, dates_df, viz_data, extra_tags, filename, scenario_images=None, flowchart_image=None, config=None, output_directory=None):
    """Run Task 3 with prompt engineering for any product type."""
    print("Running Task 3 (Monte Carlo Pricing with AI Methodology)...")

    # Extract parameters with LLM methodology
    extractor = PricingParameterExtractor(config or _Config())
    params = extractor.extract_all(key_df, dates_df, viz_data, extra_tags)

    print(f"  Product Category: {params.product_category}")
    print(f"  Features: autocall={params.has_autocall}, KI={params.has_knock_in}, KO={params.has_knock_out}, protection={params.has_capital_protection}")
    print(f"  Coupon: {params.coupon_rate:.4%} ({params.coupon_type})")
    if params.ko_barrier:
        print(f"  KO Barrier: {params.ko_barrier:.2%}")
    if params.ki_barrier:
        print(f"  KI Barrier: {params.ki_barrier:.2%}")

    # Run pricing
    pricer = UniversalMonteCarloPricer(params)
    results = pricer.price()
    greeks = pricer.calculate_greeks()
    risks = pricer.calculate_risk_metrics()

    # Save reports as ordinary files.
    html = generate_pricing_html_report(params, results, greeks, risks)

    # Generate outputs
    base_filename = os.path.splitext(filename)[0]
    html_filename = os.path.join(str(output_directory or "."), f"comprehensive_report_{base_filename}.html")
    with open(html_filename, "w", encoding="utf-8") as report:
        report.write(html)
    print(f"  HTML: {html_filename}")
    json_filename = os.path.join(str(output_directory or "."), f"comprehensive_report_{base_filename}.json")
    json_data = generate_comprehensive_json(filename, key_df, dates_df, viz_data, extra_tags, params, results, greeks, risks)
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  JSON: {json_filename}")

    pdf_filename = generate_comprehensive_pdf(filename, key_df, dates_df, viz_data, extra_tags, params, results, greeks, risks, scenario_images, flowchart_image, output_directory=output_directory)
    if pdf_filename:
        print(f"  PDF: {pdf_filename}")

    return {'params': params, 'results': results, 'greeks': greeks, 'risks': risks}
