"""Reproduce known research-model defects; not valuation validation.

Run from repository root: python -m tools.diagnose_pricing
The examples are deterministic and synthetic. See docs/CODE_REVIEW.md.
"""
import json
import numpy as np
from termsheet_analyser.step06_parameters import PricingParameters
from termsheet_analyser.step07_pricing import UniversalMonteCarloPricer


def diagnose():
    p=PricingParameters(product_category="PARTICIPATION",n_simulations=64,n_steps_per_year=12)
    engine=UniversalMonteCarloPricer(p)
    base=engine.price()["fair_value"]
    p.initial_levels*=1.1
    bumped=engine.price()["fair_value"]
    # The payoff reference is bumped with spot, so the normalisation cancels it.
    spot_invariance=abs(base-bumped)<1e-12

    values=[]
    for maturity in [1,2]:
        p=PricingParameters(product_category="AUTOCALLABLE",has_fixed_coupon=True,coupon_rate=.06,
                            maturity_years=maturity,risk_free_rate=0,n_simulations=2,n_steps_per_year=252,
                            volatilities=np.array([0.0]))
        values.append(UniversalMonteCarloPricer(p).price()["fair_value"])

    p=PricingParameters(product_category="PARTICIPATION",n_simulations=64,n_steps_per_year=12)
    engine=UniversalMonteCarloPricer(p)
    engine.price()
    original=engine.payoffs.copy()
    engine.calculate_greeks()
    stale_after_greeks=not np.allclose(original,engine.payoffs,atol=1e-12,rtol=0)
    return {
        "spot_100_price":base,"spot_110_price":bumped,
        "spot_bump_cancelled_by_moving_reference":bool(spot_invariance),
        "annual_coupon_6pct_autocall_1y_value":values[0],
        "annual_coupon_6pct_autocall_2y_value":values[1],
        "two_year_coupon_not_scaled":bool(abs(values[0]-values[1])<1e-12),
        "payoffs_not_restored_after_greeks":bool(stale_after_greeks),
    }


if __name__=="__main__":
    print(json.dumps(diagnose(),indent=2))
