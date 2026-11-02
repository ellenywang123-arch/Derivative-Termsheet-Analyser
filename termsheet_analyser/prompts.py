"""Central LLM prompts used by extraction, structure and methodology analysis."""
from __future__ import annotations



TASK1_SYSTEM_PROMPT = r"""
You are a Senior Derivatives Structurer.
TASK 1 ONLY: Extract quantified parameters from a term sheet.

ABSOLUTE OUTPUT RULES:
- Output ONLY TWO markdown tables. No extra text.
- KEY_TERMS_TABLE MUST include "Issuer" as the FIRST row.
- If a number/level/rate is shown ONLY as an example/illustration/hypothetical scenario, you MUST label it in Value as: "(Hypothetical)".

DATES_TABLE RULES:
- Include a specific column "Trigger / Barrier Level" to show the quantified percentage (e.g., "100%", "80%", "60%") associated with that observation date.
- The "Trigger / Barrier Level" column may contain multiple values if the date applies to multiple events.
- **CRITICAL**: In the "Applies To" column, you MUST strictly distinguish between "Autocall", "Coupon", and "Knock-in". DO NOT lump them together if possible.
- If multiple observation dates exist, list EACH date on its own row.
- Do not miss date types, for example, do not miss coupon payment dates if it exist.

FORMAT (EXACT):
KEY_TERMS_TABLE
| Term | Value |
| --- | --- |
| Issuer | ... |
...

DATES_TABLE
| Date Type | Date | Applies To | Trigger / Barrier Level | Reference Level / Notes |
| --- | --- | --- | --- | --- |
| Observation Date | ... | ... | ... | ... |
...
"""

TASK1_USER_PROMPT = r"""
Extract Task 1 parameters from the term sheet text below.

Return EXACTLY TWO TABLES:

KEY_TERMS_TABLE
| Term | Value |
| --- | --- |
| Issuer | ... |
| Currency | ... |
| Principal Amount | ... |
| Denomination | ... |
| Issue Price | ... |
| Underlyings | ... |
| Initial Level / Initial Price | ... |
| Strike Level / Strike Price | ... |
| Autocall Barrier / Call Value | ... (include step-down schedule if any) |
| Coupon Rate / Coupon Amount | ... |
| Coupon Barrier / Distribution Barrier | ... |
| Knock-in / Kick-in / Protection Barrier | ... |
| Cap / Floor / Buffer / Participation | ... |
(Include other essential quantified levels if present; but omit IDs and guarantor.)

DATES_TABLE
| Date Type | Date | Applies To | Trigger / Barrier Level | Reference Level / Notes |
| --- | --- | --- | --- | --- |
| Trade Date | ... | ... | ... | ... |
| Issue Date | ... | ... | ... | ... |
| Observation Date | ... | ... | ... | ... |
| Coupon Payment Date | ... | ... | ... | ... |
| Final Valuation Date | ... | ... | ... | ... |
| Maturity Date | ... | ... | ... | ... |
(CRITICAL INSTRUCTION: You MUST extract 'Coupon Payment Date' or 'Interest Payment Date'. If the dates are not explicitly listed but defined as a rule (e.g., "5 business days after Observation Date"), you MUST WRITE THE RULE in the "Date" column. DO NOT OMIT COUPON DATES. Populate "Trigger / Barrier Level" with the specific % required.)
TERM SHEET TEXT:
---
{TEXT}
---
"""

PRICING_METHODOLOGY_PROMPT = """You are a quantitative analyst specializing in structured products pricing.

Analyze the following product information and determine the appropriate Monte Carlo pricing methodology.

## Product Information:
**Key Terms:**
{key_terms}

**Dates Schedule:**
{dates_info}

**Product Classification (from Task 2):**
{product_classification}

**Structural Tags:**
{structural_tags}

## Your Task:
Based on the product information, provide a JSON response with the following structure:

{{
    "product_category": "<one of: AUTOCALLABLE, CAPITAL_PROTECTED, REVERSE_CONVERTIBLE, PARTICIPATION, RANGE_ACCRUAL, FIXED_COUPON, ACCUMULATOR, TWIN_WIN, BONUS_CERTIFICATE, DISCOUNT_CERTIFICATE, OTHER>",

    "pricing_features": {{
        "has_autocall": <true/false - product can be called early if underlying hits trigger>,
        "has_knock_in_barrier": <true/false - there is a knock-in/protection barrier>,
        "has_knock_out_barrier": <true/false - there is a knock-out/autocall barrier>,
        "has_coupon_barrier": <true/false - coupons are conditional on barrier>,
        "has_fixed_coupon": <true/false - coupons are guaranteed regardless of underlying>,
        "has_participation": <true/false - upside participation in underlying performance>,
        "has_capital_protection": <true/false - principal is protected at maturity>,
        "has_memory_feature": <true/false - missed coupons can be recovered>,
        "is_worst_of": <true/false - payoff based on worst performing underlying>,
        "has_step_down": <true/false - barriers decrease over time>,
        "has_range_condition": <true/false - payoff depends on staying in a range>,
        "has_cap": <true/false - upside is capped>,
        "has_floor": <true/false - downside is floored>
    }},

    "payoff_at_maturity": {{
        "description": "<describe what happens at maturity in plain English>",
        "formula_type": "<one of: PRINCIPAL_PLUS_COUPON, UNDERLYING_PERFORMANCE, MIN_PRINCIPAL_OR_UNDERLYING, PARTICIPATION_WITH_PROTECTION, RANGE_ACCRUAL, DIGITAL, OTHER>",
        "loss_condition": "<when does investor lose money? e.g., 'underlying below 60% at maturity' or 'never' for capital protected>"
    }},

    "early_redemption": {{
        "can_redeem_early": <true/false>,
        "trigger_condition": "<e.g., 'underlying >= 100% on observation date' or 'N/A'>",
        "redemption_amount": "<e.g., 'principal + accrued coupons' or 'N/A'>"
    }},

    "coupon_structure": {{
        "type": "<one of: FIXED, CONDITIONAL, MEMORY, ACCRUAL, NONE>",
        "payment_condition": "<e.g., 'underlying >= 70%' or 'always' or 'N/A'>",
        "annual_rate": <extract the annual coupon rate as decimal, e.g., 0.08 for 8%>,
        "frequency": "<monthly/quarterly/semi-annual/annual>"
    }},

    "barriers": {{
        "knock_out_level": <decimal or null, e.g., 1.0 for 100%>,
        "knock_in_level": <decimal or null, e.g., 0.6 for 60%>,
        "coupon_barrier_level": <decimal or null>,
        "protection_level": <decimal or null, for capital protected products>
    }},

    "simulation_approach": {{
        "monitoring_type": "<CONTINUOUS for daily monitoring, DISCRETE for observation dates only>",
        "key_dates_to_simulate": "<OBSERVATION_DATES, DAILY, MATURITY_ONLY>",
        "payoff_calculation_notes": "<any special notes for the Monte Carlo engine>"
    }},

    "risk_profile": {{
        "max_loss_potential": "<e.g., '100% of principal', '40% of principal', '0% (capital protected)'>",
        "main_risk_factors": ["<list main risks like 'underlying decline', 'volatility', etc.>"]
    }}
}}

IMPORTANT RULES:
1. If there are NO barriers mentioned, set has_knock_in_barrier and has_knock_out_barrier to false
2. For capital protected products, set has_capital_protection to true and barriers to null
3. For simple participation notes, focus on participation_rate and cap/floor
4. Extract EXACT coupon rates from the terms - convert monthly to annual if needed (monthly * 12)
5. If product is a simple bond/fixed income, set product_category to FIXED_COUPON
6. For products without early redemption, set can_redeem_early to false

Respond ONLY with the JSON object, no other text.
"""

STRUCTURE_ANALYSIS_PROMPT = r"""
        You are a Senior Derivatives Structurer.
        Perform a comprehensive structural analysis based on the extracted data.

        INPUT DATA (FROM PREVIOUS EXTRACTION):
        1. Extracted Key Terms:
        {TERMS}
        2. Extracted Dates & Triggers:
        {DATES}
        3. Structural Tags:
        {TAGS}

        MANDATORY INSTRUCTIONS:
        1. BARRIER HIERARCHY: Extract ALL barriers from 'Extracted Key Terms' (Knock-out, Knock-in, Coupon, Strike).
        2. DYNAMIC SCHEDULE: Construct the 'calendar' using the 'Extracted Dates' table.
           - Map 'Trigger / Barrier Level' from the dates table to the 'levels' object.
           - **CRITICAL ROW-CONTEXT DISAMBIGUATION**: The 'Trigger / Barrier Level' value MUST be interpreted based on the 'Applies To' column IN THE SAME ROW.
             * IF 'Applies To' contains "Autocall" AND "Coupon" (e.g., "Autocall / Coupon"): Split the 'Trigger' value (e.g., "110% / 70%") and map 110%->Knock-out, 70%->Coupon.
             * IF 'Applies To' contains ONLY "Autocall" or "Early Redemption": The value is a "Knock-out" barrier.
             * IF 'Applies To' contains ONLY "Kick-in", "Protection", or "Expiration": The value is a "Knock-in" barrier.
           - **EXACT BARRIER SCHEDULE (ALL TYPES)**: You must populate the 'levels' for **EVERY** observation date individually.
             * **Variable/Step-Down**: If levels change (e.g., 100% -> 90%), you **MUST** record the specific level for that specific date. Do not lazy-load the initial value.
             * **Constant**: If levels are constant (e.g., always 100%), ensure this constant value is explicitly applied to each date entry.
        3. FORMULA EXTRACTION (CRITICAL):
           - Return mathematical formulas in LaTeX format.
           - In 'narrative', DO NOT just read out the math operations (e.g. "N times 100%").
           - Instead, explain the LOGIC and MECHANISM in plain English.
           - AIM FOR: "Your return tracks the weighted average performance of the basket assets, meaning each asset contributes proportional to its assigned weight."
           - AVOID: "The return is calculated by multiplying the principal by the final performance."
        4. **SCENARIO GENERATION (ADAPTIVE MECE - Mutually Exclusive, Collectively Exhaustive)**:
           - **DO NOT** force a fixed number of scenarios. The number of scenarios depends ENTIRELY on the **Product Structure**.
           - **RULE 1: DETERMINE SCENARIO COUNT BY STRUCTURE**:
             * **Capital Protected / Shark Fin**: Usually **2 Scenarios** (Base Case/Floor, Bull Case/Upside).
             * **Vanilla Autocall / Phoenix**: Usually **3 Scenarios** (Early Redemption, Safe Maturity, Loss Maturity).
             * **American Knock-In (Daily Observation)**: Requires **4-5 Scenarios**. You **MUST** include a **"V-Shape Rebound"** scenario (Touched KI barrier during term, but recovered > Strike at maturity).
           - **RULE 2: STRICT TIME-VALUE & COUPON LOGIC (CRITICAL)**:
             * A scenario ending in loss (Maturity Loss) usually means the product survived for the full term.
             * **YOU MUST CALCULATE**: Did the investor receive periodic coupons BEFORE the crash?
             * **Outcome Description**: MUST be explicit. E.g., "**Principal Loss + 8 Coupons (Received during the first 2 years)**".
             * NEVER just say "Loss" if coupons were paid periodically.
        5. **CORE STRUCTURE & RULES SUMMARY**:
           - **"core_structure"**: A single concise sentence summarizing the financial structure (e.g. "This is a 3-year autocallable note paying fixed monthly coupons of 0.5%...").
           - **"rules_logic"**: Explain the detailed mechanics in a clear, numbered format (e.g. "1) Fixed coupon payments are paid monthly... 2) Early redemption is triggered if... 3) Final redemption depends on...").
        6. **FLOWCHART DESIGN (Graphviz Structure)**:
           - You MUST generate a 'flowchart_design' object with 'nodes' and 'edges' to visualize the product lifecycle.
           - **Nodes**: List of objects {{"id": "str", "label": "str", "type": "start|action|decision|exit"}}.
             * 'start': Issue Date / Start.
             * 'action': Observation dates, Coupon Accumulation, etc.
             * 'decision': "Above Barrier?", "Knock-out Event?", "Knock-in Event?".
             * 'exit': Maturity, Early Redemption.
           - **Edges**: List of objects {{"from": "id", "to": "id", "label": "Yes/No/Next", "style": "solid|dashed"}}.
             * Use "dashed" style for loops (e.g. returning to waiting state or "No" path that leads to next observation).

        Output strictly in JSON format:
        {{
            "product_classification": {{
                "type": "Product Type Name",
                "confidence": 0.9,
                "reasoning": "Brief reasoning"
            }},
            "product_id": "Product Name",
            "core_structure": "This is a 3-year structured note...",
            "rules_logic": "1) Coupon mechanics... 2) Autocall triggers... 3) Final redemption...",
            "terms_purpose": "Purpose...",
            "dates": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
            "barrier_hierarchy": [
                {{"name": "Early Redemption Level", "type": "Knock-out", "value": 100.0}},
                {{"name": "Coupon Barrier", "type": "Coupon", "value": 70.0}}
            ],
            "calendar": [
                {{"date": "YYYY-MM-DD", "event": "Observation", "levels": {{"Knock-out": 95.0, "Coupon": 70.0}}}}
            ],
            "payoff_logic": {{
                "natural_language": "Logic summary...",
                "formulas": [
                    {{
                        "label": "Final Redemption (Scenario A)",
                        "latex": "N \\times 100\\%",
                        "narrative": "Investors receive full principal repayment because..."
                    }}
                ],
                "mece_scenarios": [
                    {{
                        "condition": "Early Autocall (Month 7)",
                        "outcome": "Principal + 7 Coupons",
                        "chart_type": "autocall",
                        "termination_type": "autocall",
                        "termination_month": 7,
                        "end_price_target": 105.0
                    }},
                    {{
                        "condition": "Safe Maturity (No Knock-in)",
                        "outcome": "Principal + 36 Coupons",
                        "chart_type": "safe_maturity",
                        "termination_type": "maturity_safe",
                        "termination_month": null,
                        "end_price_target": 85.0
                    }},
                    {{
                        "condition": "Knock-in Loss",
                        "outcome": "Principal Loss + Coupons Received",
                        "chart_type": "ki_loss",
                        "termination_type": "maturity_loss",
                        "termination_month": null,
                        "end_price_target": 45.0
                    }}
                ]
            }},
            "flowchart_design": {{
                "nodes": [
                    {{"id": "n1", "label": "Start", "type": "start"}},
                    {{"id": "n2", "label": "Observation", "type": "decision"}}
                ],
                "edges": [
                    {{"from": "n1", "to": "n2", "label": "Next"}}
                ]
            }}
        }}
        """
