"""06 — Build experimental pricing parameters from product terms."""
from __future__ import annotations

from .prompts import PRICING_METHODOLOGY_PROMPT
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from openai import OpenAI
import json
import numpy as np
import re

class PricingMethodologyAnalyzer:
    """Uses LLM to determine appropriate pricing methodology."""

    def __init__(self, config):
        self.config = config

    def analyze(self, key_df, dates_df, viz_data, extra_tags) -> Dict:
        """Analyze product and return pricing methodology."""

        # Format key terms
        key_terms_str = ""
        if key_df is not None and not key_df.empty:
            for _, row in key_df.iterrows():
                key_terms_str += f"- {row.get('Term', '')}: {row.get('Value', '')}\n"

        # Format dates
        dates_str = ""
        if dates_df is not None and not dates_df.empty:
            for _, row in dates_df.iterrows():
                dates_str += f"- {row.get('Date Type', '')}: {row.get('Dates', row.get('Date', ''))}\n"

        # Format classification
        classification_str = ""
        if viz_data:
            prod_class = viz_data.get('product_classification', {})
            classification_str = f"Type: {prod_class.get('type', 'Unknown')}\n"
            classification_str += f"Reasoning: {prod_class.get('reasoning', 'N/A')}\n"
            classification_str += f"Core Structure: {viz_data.get('core_structure', 'N/A')}"

        # Format tags
        tags_str = json.dumps(extra_tags, indent=2) if extra_tags else "{}"

        # Build prompt
        prompt = PRICING_METHODOLOGY_PROMPT.format(
            key_terms=key_terms_str or "No key terms extracted",
            dates_info=dates_str or "No dates extracted",
            product_classification=classification_str or "No classification available",
            structural_tags=tags_str
        )

        try:
            client = OpenAI(api_key=self.config.API_KEY, base_url=self.config.BASE_URL)
            response = client.chat.completions.create(
                model=self.config.MODEL,
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = (response.choices[0].message.content or "").strip()

            # Clean JSON
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)

            methodology = json.loads(response_text)
            supported = {"AUTOCALLABLE", "CAPITAL_PROTECTED", "PARTICIPATION", "FIXED_COUPON", "REVERSE_CONVERTIBLE", "RANGE_ACCRUAL", "BONUS_CERTIFICATE", "DISCOUNT_CERTIFICATE"}
            if not isinstance(methodology, dict) or methodology.get("product_category") not in supported:
                raise ValueError("Missing or unsupported product category.")
            if not isinstance(methodology.get("pricing_features"), dict):
                raise ValueError("Pricing features must be a JSON object.")
            return methodology

        except Exception as e:
            raise ValueError("Pricing methodology could not be established; pricing stopped.") from e

    def _get_default_methodology(self) -> Dict:
        """Legacy example only. Never used automatically when analysis fails."""
        return {
            "product_category": "AUTOCALLABLE",
            "pricing_features": {
                "has_autocall": True,
                "has_knock_in_barrier": True,
                "has_knock_out_barrier": True,
                "has_coupon_barrier": True,
                "has_fixed_coupon": False,
                "has_participation": False,
                "has_capital_protection": False,
                "has_memory_feature": False,
                "is_worst_of": False,
                "has_step_down": False,
                "has_range_condition": False,
                "has_cap": False,
                "has_floor": False
            },
            "payoff_at_maturity": {
                "description": "Principal returned if no knock-in, otherwise linked to underlying",
                "formula_type": "MIN_PRINCIPAL_OR_UNDERLYING",
                "loss_condition": "underlying below knock-in at any time AND below 100% at maturity"
            },
            "early_redemption": {
                "can_redeem_early": True,
                "trigger_condition": "underlying >= 100% on observation date",
                "redemption_amount": "principal + accrued coupons"
            },
            "coupon_structure": {
                "type": "CONDITIONAL",
                "payment_condition": "underlying >= coupon barrier",
                "annual_rate": 0.06,
                "frequency": "monthly"
            },
            "barriers": {
                "knock_out_level": 1.0,
                "knock_in_level": 0.6,
                "coupon_barrier_level": 0.7,
                "protection_level": None
            },
            "simulation_approach": {
                "monitoring_type": "CONTINUOUS",
                "key_dates_to_simulate": "OBSERVATION_DATES",
                "payoff_calculation_notes": "Standard autocallable with barrier monitoring"
            },
            "risk_profile": {
                "max_loss_potential": "100% of principal",
                "main_risk_factors": ["underlying decline", "knock-in breach"]
            }
        }

@dataclass
class PricingParameters:
    """Pricing parameters for all product types."""
    # Basic info
    notional: float = 1.0  # <--- CHANGED THIS TO 1.0
    currency: str = "USD"
    product_category: str = "AUTOCALLABLE"

    # Underlyings
    n_underlyings: int = 1
    underlying_names: List[str] = field(default_factory=list)
    initial_levels: np.ndarray = field(default_factory=lambda: np.array([100.0]))

    # Barriers (None if not applicable)
    ko_barrier: float = None
    ki_barrier: float = None
    coupon_barrier: float = None
    protection_level: float = None
    ko_schedule: List[Tuple[int, float]] = field(default_factory=list)

    # Coupon
    coupon_rate: float = 0.0
    coupon_type: str = "CONDITIONAL"  # FIXED, CONDITIONAL, MEMORY, ACCRUAL, NONE
    coupon_frequency: str = "monthly"

    # Product features (from prompt engineering)
    has_autocall: bool = False
    has_knock_in: bool = False
    has_knock_out: bool = False
    has_coupon_barrier: bool = False
    has_fixed_coupon: bool = False
    has_participation: bool = False
    has_capital_protection: bool = False
    has_memory_feature: bool = False
    is_worst_of: bool = False
    has_step_down: bool = False
    has_range_condition: bool = False
    has_cap: bool = False
    has_floor: bool = False

    # Participation
    participation_rate: float = 1.0
    cap_level: float = None
    floor_level: float = None

    # Range (for range accrual)
    range_lower: float = None
    range_upper: float = None

    # Market assumptions
    risk_free_rate: float = 0.03
    dividend_yields: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    volatilities: np.ndarray = field(default_factory=lambda: np.array([0.25]))
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))

    # Timing
    maturity_years: float = 1.0
    observation_dates: List[str] = field(default_factory=list)

    # Simulation
    n_simulations: int = 50000
    n_steps_per_year: int = 252
    monitoring_type: str = "CONTINUOUS"

    # Raw values for validation
    raw_coupon_rate: str = ""
    raw_ko_barrier: str = ""
    raw_ki_barrier: str = ""

    # Methodology from LLM
    pricing_methodology: Dict = field(default_factory=dict)

class PricingParameterExtractor:
    """Extract parameters using both rule-based and LLM analysis."""

    def __init__(self, config=None):
        self.config = config
        self.params = PricingParameters()
        self.extraction_log = []

    def extract_all(self, key_df, dates_df, viz_data, extra_tags) -> PricingParameters:
        """Extract all parameters using LLM methodology + rule-based extraction."""

        # Each extraction starts afresh; schedules must not leak between documents.
        self.params = PricingParameters()
        self.extraction_log = []
        # Step 1: Get pricing methodology from LLM
        if self.config:
            analyzer = PricingMethodologyAnalyzer(self.config)
            methodology = analyzer.analyze(key_df, dates_df, viz_data, extra_tags)
            self.params.pricing_methodology = methodology
            self._apply_methodology(methodology)

        # Step 2: Rule-based extraction (can override/supplement LLM)
        self._extract_from_key_terms(key_df)
        self._extract_from_dates(dates_df)
        self._extract_from_viz_data(viz_data)
        self._extract_from_tags(extra_tags)

        # Step 3: Finalize
        self._finalize_params()

        return self.params

    def _apply_methodology(self, methodology: Dict) -> None:
        """Apply LLM-determined methodology to parameters."""

        # Product category
        self.params.product_category = methodology.get('product_category', 'AUTOCALLABLE')

        # Pricing features
        features = methodology.get('pricing_features', {})
        self.params.has_autocall = features.get('has_autocall', False)
        self.params.has_knock_in = features.get('has_knock_in_barrier', False)
        self.params.has_knock_out = features.get('has_knock_out_barrier', False)
        self.params.has_coupon_barrier = features.get('has_coupon_barrier', False)
        self.params.has_fixed_coupon = features.get('has_fixed_coupon', False)
        self.params.has_participation = features.get('has_participation', False)
        self.params.has_capital_protection = features.get('has_capital_protection', False)
        self.params.has_memory_feature = features.get('has_memory_feature', False)
        self.params.is_worst_of = features.get('is_worst_of', False)
        self.params.has_step_down = features.get('has_step_down', False)
        self.params.has_range_condition = features.get('has_range_condition', False)
        self.params.has_cap = features.get('has_cap', False)
        self.params.has_floor = features.get('has_floor', False)

        # Barriers from methodology
        barriers = methodology.get('barriers', {})
        if barriers.get('knock_out_level') is not None:
            self.params.ko_barrier = barriers['knock_out_level']
        if barriers.get('knock_in_level') is not None:
            self.params.ki_barrier = barriers['knock_in_level']
        if barriers.get('coupon_barrier_level') is not None:
            self.params.coupon_barrier = barriers['coupon_barrier_level']
        if barriers.get('protection_level') is not None:
            self.params.protection_level = barriers['protection_level']

        # Coupon from methodology
        coupon = methodology.get('coupon_structure', {})
        self.params.coupon_type = coupon.get('type', 'CONDITIONAL')
        if coupon.get('annual_rate') is not None:
            self.params.coupon_rate = coupon['annual_rate']
        if coupon.get('frequency'):
            self.params.coupon_frequency = coupon['frequency']

        # Simulation approach
        sim = methodology.get('simulation_approach', {})
        self.params.monitoring_type = sim.get('monitoring_type', 'CONTINUOUS')

    def _extract_from_key_terms(self, key_df) -> None:
        """Extract from Task 1 key terms (supplements/overrides LLM)."""
        if key_df is None or key_df.empty:
            return

        for _, row in key_df.iterrows():
            term = str(row.get('Term', '')).strip()
            term_lower = term.lower()
            value = str(row.get('Value', '')).strip()

            # Notional
            if any(kw in term_lower for kw in ['principal', 'notional', 'nominal', 'denomination']):
                parsed = self._parse_number(value)
                if 0:
                    self.params.notional = parsed

            # Currency
            if 'currency' in term_lower:
                self.params.currency = self._extract_currency(value)

            # Underlyings
            if 'underlying' in term_lower:
                parts = re.split(r'[,;/&]|\band\b', value)
                names = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
                if names:
                    self.params.n_underlyings = len(names)
                    self.params.underlying_names = names
                    self.params.initial_levels = np.array([100.0] * len(names))
                    self.params.volatilities = np.array([0.25] * len(names))
                    self.params.dividend_yields = np.array([0.0] * len(names))

            # KO Barrier - only if product has autocall
            if self.params.has_autocall or self.params.has_knock_out:
                if any(kw in term_lower for kw in ['autocall', 'knock-out', 'call trigger', 'early redemption trigger']):
                    pcts = self._extract_percentages(value)
                    if pcts:
                        self.params.ko_barrier = pcts[0]
                        self.params.raw_ko_barrier = value

            # KI Barrier - only if product has knock-in
            if self.params.has_knock_in:
                if any(kw in term_lower for kw in ['knock-in', 'kick-in', 'barrier level', 'protection barrier']):
                    if 'coupon' not in term_lower:
                        pcts = self._extract_percentages(value)
                        if pcts:
                            self.params.ki_barrier = pcts[0]
                            self.params.raw_ki_barrier = value

            # Coupon barrier
            if self.params.has_coupon_barrier:
                if 'coupon barrier' in term_lower or 'distribution barrier' in term_lower:
                    pcts = self._extract_percentages(value)
                    if pcts:
                        self.params.coupon_barrier = pcts[0]

            # Coupon rate
            if any(kw in term_lower for kw in ['coupon', 'interest', 'yield']) and 'barrier' not in term_lower:
                rate = self._parse_coupon_rate(value)
                if rate is not None:
                    self.params.coupon_rate = rate
                    self.params.raw_coupon_rate = value

            # Participation
            if 'participation' in term_lower:
                pcts = self._extract_percentages(value)
                if pcts:
                    self.params.participation_rate = pcts[0]
                    self.params.has_participation = True

            # Cap
            if 'cap' in term_lower and 'floor' not in term_lower:
                pcts = self._extract_percentages(value)
                if pcts:
                    self.params.cap_level = pcts[0]
                    self.params.has_cap = True

            # Floor
            if 'floor' in term_lower or 'minimum return' in term_lower:
                pcts = self._extract_percentages(value)
                if pcts:
                    self.params.floor_level = pcts[0]
                    self.params.has_floor = True

            # Protection level (capital protected)
            if 'protection' in term_lower and 'barrier' not in term_lower:
                pcts = self._extract_percentages(value)
                if pcts:
                    self.params.protection_level = pcts[0]
                    self.params.has_capital_protection = True

    def _extract_from_dates(self, dates_df) -> None:
        """Extract dates from Task 1."""
        if dates_df is None or dates_df.empty:
            return

        issue_date = None
        maturity_date = None

        for _, row in dates_df.iterrows():
            date_type = str(row.get('Date Type', '')).lower()
            date_val = str(row.get('Dates', row.get('Date', '')))
            trigger = str(row.get('Trigger / Barrier Level', ''))

            explicit_dates = self._extract_explicit_dates(date_val)

            if 'issue' in date_type and 'maturity' not in date_type:
                if explicit_dates:
                    issue_date = explicit_dates[0]

            elif 'maturity' in date_type or 'final' in date_type:
                if explicit_dates:
                    maturity_date = explicit_dates[-1]

            elif 'observation' in date_type or 'valuation' in date_type:
                barriers = self._extract_percentages(trigger)
                for i, d in enumerate(explicit_dates):
                    self.params.observation_dates.append(d.strftime('%Y-%m-%d'))
                    if self.params.has_autocall and issue_date:
                        barrier = barriers[i] if i < len(barriers) else (self.params.ko_barrier or 1.0)
                        days = (d - issue_date).days
                        if days > 0:
                            self.params.ko_schedule.append((days, barrier))

        if issue_date and maturity_date:
            self.params.maturity_years = (maturity_date - issue_date).days / 365.0

        if self.params.ko_schedule:
            self.params.ko_schedule.sort(key=lambda x: x[0])

    def _extract_from_viz_data(self, viz_data: Dict) -> None:
        """Extract from Task 2 analysis."""
        if not viz_data:
            return

        # Only fill in gaps, don't override
        if self.params.maturity_years == 1.0:
            dates_info = viz_data.get('dates', {})
            if dates_info:
                try:
                    start = datetime.strptime(dates_info.get('start', ''), '%Y-%m-%d')
                    end = datetime.strptime(dates_info.get('end', ''), '%Y-%m-%d')
                    self.params.maturity_years = (end - start).days / 365.0
                except:
                    pass

    def _extract_from_tags(self, tags: Dict) -> None:
        """Extract from structural tags."""
        if not tags:
            return

        if tags.get('is_memory'):
            self.params.has_memory_feature = True
        if tags.get('is_worst_of'):
            self.params.is_worst_of = True

    def _finalize_params(self) -> None:
        """Finalize parameters."""
        n = self.params.n_underlyings
        if n > 1:
            corr = np.full((n, n), 0.5)
            np.fill_diagonal(corr, 1.0)
            self.params.correlation_matrix = corr
        else:
            self.params.correlation_matrix = np.array([[1.0]])

    def _extract_explicit_dates(self, text: str) -> List[datetime]:
        """Extract only explicit calendar dates."""
        dates = []
        text = text.replace('<br>', ' ').replace('\n', ' ')

        # YYYY-MM-DD
        for m in re.finditer(r'(\d{4})-(\d{2})-(\d{2})', text):
            try:
                dates.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except:
                pass

        # Month DD, YYYY
        for m in re.finditer(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', text, re.I):
            try:
                month = datetime.strptime(m.group(1), '%B').month
                dates.append(datetime(int(m.group(3)), month, int(m.group(2))))
            except:
                pass

        # DD Month YYYY
        for m in re.finditer(r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', text, re.I):
            try:
                month = datetime.strptime(m.group(2), '%B').month
                dates.append(datetime(int(m.group(3)), month, int(m.group(1))))
            except:
                pass

        seen = set()
        unique = []
        for d in dates:
            key = d.strftime('%Y-%m-%d')
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return sorted(unique)

    def _extract_percentages(self, text: str) -> List[float]:
        """Extract percentages as decimals."""
        pcts = []
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text):
            val = float(m.group(1))
            pcts.append(val / 100.0)
        return pcts

    def _parse_coupon_rate(self, text: str) -> Optional[float]:
        """Parse coupon rate, handling per-period rates."""
        text_lower = text.lower()

        # Per period
        match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:per|each)?\s*(month|quarter|year|annum|p\.?a\.?)', text_lower)
        if match:
            rate = float(match.group(1))
            rate = rate / 100.0
            period = match.group(2)
            if 'month' in period:
                rate *= 12
            elif 'quarter' in period:
                rate *= 4
            return rate

        # Simple percentage
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if match:
            rate = float(match.group(1))
            rate = rate / 100.0
            if rate < 0.03 and 'month' in text_lower:
                rate *= 12
            return rate
        return None

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse number from text."""
        try:
            clean = re.sub(r'[^\d.\-]', '', text.replace(',', ''))
            return float(clean) if clean else None
        except:
            return None

    def _extract_currency(self, text: str) -> str:
        """Extract currency code."""
        for c in ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'HKD', 'SGD', 'AUD', 'CHF', 'CAD']:
            if c in text.upper():
                return c
        return 'USD'
