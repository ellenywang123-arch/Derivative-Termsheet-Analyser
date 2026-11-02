"""03 — Interpret product structure using the extracted evidence."""
from __future__ import annotations

from .prompts import STRUCTURE_ANALYSIS_PROMPT
from openai import OpenAI
import json
import re
import time

class LogicArchitect:
    """Task 2 Logic: Analyzes structure using Task 1 Inputs + Text."""
    def __init__(self, config, client=None):
        self.client = client if client is not None else OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.model = config.MODEL

    def analyze_structure(self, text, key_df=None, dates_df=None, extra_tags=None):
        terms_context = key_df.to_string() if key_df is not None and not key_df.empty else "No Key Terms extracted."
        dates_context = dates_df.to_string() if dates_df is not None and not dates_df.empty else "No Dates extracted."
        tags_context = json.dumps(extra_tags) if extra_tags else "No tags."

        prompt = STRUCTURE_ANALYSIS_PROMPT.format(TERMS=terms_context, DATES=dates_context, TAGS=tags_context)

        max_retries = 3
        viz_data = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                    temperature=0.1
                )
                content = response.choices[0].message.content
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_match:
                    viz_data = json.loads(json_match.group(1).strip())
                    break
                elif "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                    viz_data = json.loads(content.strip())
                    break
            except Exception:
                time.sleep(1)

        return viz_data
