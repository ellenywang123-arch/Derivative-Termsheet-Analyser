# Code review of the supplied notebook

This review distinguishes the notebook's original research limitations from defects introduced by the earlier source packaging. It is not a certification of financial-model correctness. The API was mocked for tests; no real credential or confidential PDF was used. Reproduction commands below run on synthetic inputs.

## Fixed in this refactor

| Problem | Origin | Change |
| --- | --- | --- |
| `PricingParameters` lost `@dataclass`; mutable fields became `Field` objects | Earlier packaging, not the original notebook | Restore the decorator; test independent mutable defaults and keyword construction |
| Notebook adapter calls `Image` without importing it | Earlier packaging | Import the correct IPython display object and use `filename=` |
| Methodology analysis calls `config.client.messages.create` and `config.model`, incompatible with the uppercase OpenAI-compatible config | Original notebook | Use `OpenAI(...).chat.completions.create` and `config.MODEL` consistently |
| API/JSON failures silently substitute an invented autocallable product with default barriers/coupon | Original notebook | Raise an error; reject missing/unsupported product categories |
| `0.5%` becomes `0.5` rather than `0.005`, making a `0.5% per month` coupon 600% annual rather than 6% | Original notebook | Explicit `%` values always divide by 100; regression tests cover 0.5%, 1% and 60% |
| Reusing an extractor accumulates state from the previous document | Original notebook | Reset parameters and extraction log at the start of each extraction |
| Installing IPython is mistaken for actually running inside a notebook; `display_id` can return no handle in a terminal | Original notebook | Isolate notebook display from computation and remove typewriter sleeps |
| Scenario seed uses Python's process-randomised `hash()` | Original notebook | Derive a stable SHA-256 seed; this still uses NumPy global state, so concurrent execution remains a limitation |
| Compressed date tables merge different events and lose row-level `Applies To` association before subsequent analysis | Original workflow | The new workflow passes and saves original row-level date records; compression remains a presentation helper |
| Scenarios are displayed but their filenames are not collected for later reporting | Original notebook orchestration | Collect filenames and save new workflow artifacts to the selected output directory |
| A long PDF is silently truncated to 70,000 characters | Original helper | New workflow detects overflow and stops rather than analysing a partial document silently |
| A reporting function installs ReportLab at runtime | Original notebook | Declare dependencies in requirements; no runtime pip installation |
| Prompt shows malformed JSON and contradictory barrier disambiguation instructions | Original notebook | Correct the example comma and the Autocall/Coupon wording |

The original extraction, chart, payoff, export functions and prompts have been moved, not replaced with a small demonstration. See `source-map.json` for original source line ranges and destinations. The legacy notebook adapter is retained separately from the new workflow entry point.

## Remaining model problems — do not present these outputs as validated valuations

### P1: Spot Greeks change the payoff reference together with spot

Location: `step07_pricing.py`, `_generate_paths`, `_get_performance`, `calculate_greeks`.

Paths start at `params.initial_levels`, and performance is divided by that same field. Bumping `initial_levels` therefore bumps both the current spot and the contractual reference. With a fixed random seed the normalised paths, and hence the value, remain invariant. An actual Delta requires separate current spot and fixed contractual reference/strike. In addition, Greek up/down simulations currently use different random draws, adding avoidable Monte Carlo noise.

Reproduction: `python -m tools.diagnose_pricing` produces identical participation values for initial levels 100 and 110 (about 0.98311948 for the synthetic 64-path example). This is a diagnostic of the implementation, not a market price.

### P1: Autocall coupon accrual does not scale with tenor or payment dates

Location: `_payoff_autocallable`.

The code divides the annual coupon by the total number of observations, so the total coupon over the whole product is one annual coupon irrespective of maturity. A deterministic, zero-rate, zero-volatility example with fixed 6% annual coupons produces 1.06 for both one and two years; under a simple non-autocalling fixed-coupon interpretation the two-year undiscounted amount would be 1.12. Other payoff branches multiply the coupon by maturity, so branches disagree.

Coupons are also accumulated and discounted at autocall/maturity rather than their individual payment dates. Current-period and memory-coupon payments can be made unconditionally on autocall or at maturity. Those rules must instead come from the actual product contract.

### P1: Calendar days and simulation steps are mixed

Location: `_payoff_autocallable`, `_extract_from_dates`.

The fallback schedule uses 21/63/126/252 as trading-step intervals and subsequently converts them as if they were calendar days using `252/365`. For a one-year monthly example, the final nominal 252-step observation is mapped to step 173 rather than step 252. Explicit schedules can also include coupon observation dates as KO dates without distinguishing event type. In the old extractor, observing rows before the issue-date row can change the resulting KO schedule. Payment dates, business-day calendars, day-count conventions and separate event schedules need an explicit model.

### P1: Greeks leave cached payoffs/statistics from a bumped valuation

Location: `calculate_greeks`, `calculate_risk_metrics`.

At the end of Greeks, parameters are restored and new paths are generated, but cached payoffs and statistics are not recomputed. The workflow calls risk metrics next, so they use the final rho-down payoff vector. `python -m tools.diagnose_pricing` demonstrates that the payoff vector differs from the original base valuation after Greeks. A robust implementation needs immutable valuation inputs, common random numbers, and either isolated pricers or explicit cache invalidation/restoration.

### P1: Market inputs and reported risk measures do not support the claimed interpretation

Location: `PricingParameters`, `_extract_from_key_terms`, `_finalize_params`, `price`, `calculate_risk_metrics`, reports.

Volatility 25%, rate 3%, dividend yield 0 and pairwise correlation 0.5 are assumptions, not observed/calibrated market data. The simulation uses a risk-neutral-style drift, while the report labels a present-value ratio as `expected_return` and terminal discounted payoff quantiles as VaR/ES. Those are not automatically a real-world expected investment return or conventional holding-period market-risk VaR. The old automatic ATTRACTIVE/NEUTRAL recommendation labels have been replaced by an unvalidated-research status. The underlying metrics retain their original names for compatibility and still require redesign.

### P2: Monitoring and product generality are overstated

Location: `_payoff_autocallable`, `_get_performance`, payoff dispatch.

The `monitoring_type` parameter is not used to change KI monitoring. A full-path minimum can record KI after the product already autocalled. Discrete GBM steps do not establish continuous barrier crossing without a bridge/other treatment. A multi-underlying product uses the first asset unless `is_worst_of` is true; weighted-basket/best-of semantics are absent. An invalid correlation matrix silently falls back to independent assets. The class name `UniversalMonteCarloPricer` is inherited; it should not be read as proof of universal product support.

### P2: Date repair and summarisation can manufacture or misassociate events

Location: `step02_extraction.py`, `enforce_observation_dates`, `compress_dates_table`.

The repair helper substitutes *all* explicit dates in the document when an observation row only states a frequency; these may include issue, payment or maturity dates. Compression groups only by Date Type, combining coupon and KO events. The new machine workflow does not use these helpers before analysis; they remain available to the old notebook presentation, where these risks are still present.

### P2: LLM output validation is partial

Location: `step03_structure.py`, `step06_parameters.py`.

The structure parser uses a greedy brace regex and retries broad exceptions, ultimately returning `None`. The new workflow checks the basic output shape, and methodology checks the category/features object; neither establishes that formulas, percentages, dates or classifications agree with the PDF. The hints tagger can match negated text such as “no autocall”. Confidence is model-generated, not a calibrated accuracy estimate. Add a product schema, source/page evidence, field validation and explicit review states before relying on it.

### P2: Units, resource use and reporting need further work

The notional extraction branch is deliberately disabled with `if 0`, leaving values normalised to 1 while reports display a currency label. Document and separate per-unit and monetary units. A 50,000-path, multi-year, multi-asset run allocates several large path/random arrays; no resource guard exists. HTML/ReportLab fields are interpolated without systematic escaping. The PDF export accepts scenario/flowchart arguments but its abbreviated body does not embed them, so it is not a complete visual report.

## Suggested repair order

1. Specify a single supported product schema and a human-reviewed, source-linked term sheet fixture.
2. Define separate spot/reference fields and coupon/KO/KI/payment event schedules with consistent units and calendars.
3. Validate payoff cashflows against hand-computed deterministic cases before Monte Carlo.
4. Isolate simulations, fix Greek bump/cache handling and validate against a reference instrument.
5. Supply reviewed market inputs; define the probability measure, risk horizon and reporting units.
6. Only then extend product types and expose numerical results to external readers.

No API keys were used during this review. Credentials from the original notebook remain excluded from this repository; rotating the old credentials is a separate account action.
