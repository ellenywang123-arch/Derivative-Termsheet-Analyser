"""07 — Experimental Monte Carlo payoffs, Greeks and risk summaries. See docs/CODE_REVIEW.md."""
from __future__ import annotations

from .step06_parameters import PricingParameters
import numpy as np

class UniversalMonteCarloPricer:
    """Monte Carlo pricer that adapts to any product type."""

    def __init__(self, params: PricingParameters):
        self.params = params
        self.paths = None
        self.payoffs = None
        self.statistics = {}

    def _generate_paths(self):
        """Generate correlated price paths."""
        n_sims = self.params.n_simulations
        n_steps = int(self.params.maturity_years * self.params.n_steps_per_year)
        n_assets = self.params.n_underlyings
        dt = 1.0 / self.params.n_steps_per_year

        try:
            L = np.linalg.cholesky(self.params.correlation_matrix)
        except:
            L = np.eye(n_assets)

        Z = np.random.standard_normal((n_sims, n_steps, n_assets))
        Z_corr = np.einsum('ijk,lk->ijl', Z, L)

        paths = np.zeros((n_sims, n_steps + 1, n_assets))
        paths[:, 0, :] = self.params.initial_levels

        for i in range(n_assets):
            vol = self.params.volatilities[i]
            r = self.params.risk_free_rate
            q = self.params.dividend_yields[i]
            drift = (r - q - 0.5 * vol**2) * dt
            diffusion = vol * np.sqrt(dt)

            for t in range(n_steps):
                paths[:, t+1, i] = paths[:, t, i] * np.exp(drift + diffusion * Z_corr[:, t, i])

        self.paths = paths
        return paths

    def _get_performance(self):
        """Get performance measure (worst-of or single)."""
        normalized = self.paths / self.params.initial_levels
        if self.params.is_worst_of and self.params.n_underlyings > 1:
            return np.min(normalized, axis=2)
        return normalized[:, :, 0]

    def _calculate_payoffs(self):
        """Calculate payoffs based on product category."""
        category = self.params.product_category

        if category == 'CAPITAL_PROTECTED':
            return self._payoff_capital_protected()
        elif category == 'PARTICIPATION':
            return self._payoff_participation()
        elif category == 'FIXED_COUPON':
            return self._payoff_fixed_coupon()
        elif category == 'REVERSE_CONVERTIBLE':
            return self._payoff_reverse_convertible()
        elif category == 'RANGE_ACCRUAL':
            return self._payoff_range_accrual()
        elif category == 'BONUS_CERTIFICATE':
            return self._payoff_bonus_certificate()
        elif category == 'DISCOUNT_CERTIFICATE':
            return self._payoff_discount_certificate()
        elif category == 'AUTOCALLABLE':
            return self._payoff_autocallable()
        raise ValueError(f"Unsupported product category: {category}")

    def _payoff_autocallable(self):
        """Standard autocallable with optional barriers."""
        n_sims = self.params.n_simulations
        n_steps = self.paths.shape[1] - 1
        performance = self._get_performance()

        payoffs = np.zeros(n_sims)
        autocalled = np.zeros(n_sims, dtype=bool)
        ki_triggered = np.zeros(n_sims, dtype=bool)
        coupons = np.zeros(n_sims)
        missed_coupons = np.zeros(n_sims)
        autocall_times = np.zeros(n_sims)

        # Observation schedule
        if self.params.ko_schedule:
            obs_days = [d for d, _ in self.params.ko_schedule]
        else:
            freq_map = {'monthly': 21, 'quarterly': 63, 'semi-annual': 126, 'annual': 252}
            interval = freq_map.get(self.params.coupon_frequency, 21)
            obs_days = list(range(interval, n_steps + 1, interval))

        obs_steps = [min(int(d * self.params.n_steps_per_year / 365), n_steps) for d in obs_days]
        obs_steps = [s for s in obs_steps if 0 < s <= n_steps]
        n_periods = max(len(obs_steps), 1)
        coupon_per_period = self.params.coupon_rate / n_periods

        # Check knock-in (if applicable)
        if self.params.has_knock_in and self.params.ki_barrier is not None:
            for sim in range(n_sims):
                if np.min(performance[sim, :]) <= self.params.ki_barrier:
                    ki_triggered[sim] = True

        # Process observations
        for obs_idx, step in enumerate(obs_steps):
            day = int(step * 365 / self.params.n_steps_per_year)

            # Get applicable KO barrier
            ko_barrier = self.params.ko_barrier or 1.0
            if self.params.ko_schedule:
                for obs_day, obs_barrier in self.params.ko_schedule:
                    if day >= obs_day:
                        ko_barrier = obs_barrier

            for sim in range(n_sims):
                if autocalled[sim]:
                    continue

                perf = performance[sim, step]

                # Autocall check
                if self.params.has_autocall and perf >= ko_barrier:
                    autocalled[sim] = True
                    autocall_times[sim] = obs_idx + 1
                    time_frac = (obs_idx + 1) / n_periods
                    disc = np.exp(-self.params.risk_free_rate * time_frac * self.params.maturity_years)
                    total_coupon = coupons[sim] + coupon_per_period
                    if self.params.has_memory_feature:
                        total_coupon += missed_coupons[sim]
                    payoffs[sim] = (1.0 + total_coupon) * disc

                # Coupon check
                elif self.params.has_fixed_coupon:
                    coupons[sim] += coupon_per_period
                elif self.params.has_coupon_barrier and self.params.coupon_barrier is not None:
                    if perf >= self.params.coupon_barrier:
                        coupons[sim] += coupon_per_period
                        if self.params.has_memory_feature:
                            coupons[sim] += missed_coupons[sim]
                            missed_coupons[sim] = 0
                    elif self.params.has_memory_feature:
                        missed_coupons[sim] += coupon_per_period
                else:
                    coupons[sim] += coupon_per_period

        # Final redemption
        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        for sim in range(n_sims):
            if autocalled[sim]:
                continue

            final_perf = performance[sim, -1]

            if ki_triggered[sim] and final_perf < 1.0:
                loss = 1.0 - final_perf
                if self.params.has_floor and self.params.floor_level is not None:
                    loss = min(loss, 1.0 - self.params.floor_level)
                payoffs[sim] = (1.0 - loss + coupons[sim]) * disc
            else:
                total_coupon = coupons[sim]
                if self.params.has_memory_feature:
                    total_coupon += missed_coupons[sim]
                payoffs[sim] = (1.0 + total_coupon) * disc

        self.statistics = {
            'prob_autocall': np.mean(autocalled),
            'prob_ki': np.mean(ki_triggered),
            'prob_loss': np.mean((ki_triggered) & (~autocalled) & (performance[:, -1] < 1.0)),
            'avg_coupons': np.mean(coupons),
            'avg_autocall_period': np.mean(autocall_times[autocalled]) if np.any(autocalled) else 0,
            'n_observations': n_periods
        }

        self.payoffs = payoffs * self.params.notional
        return self.payoffs

    def _payoff_capital_protected(self):
        """Capital protected note - principal guaranteed."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]

        protection = self.params.protection_level or 1.0
        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            perf = final_perf[sim]
            base = protection

            # Add coupon if any
            if self.params.coupon_rate > 0:
                base += self.params.coupon_rate * self.params.maturity_years

            # Add participation if applicable
            if self.params.has_participation and perf > 1.0:
                upside = (perf - 1.0) * self.params.participation_rate
                if self.params.has_cap and self.params.cap_level:
                    upside = min(upside, self.params.cap_level - 1.0)
                base += upside

            payoffs[sim] = max(protection, base)

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': 0,
            'avg_coupons': self.params.coupon_rate * self.params.maturity_years,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_participation(self):
        """Participation certificate - track underlying with leverage."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]

        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            perf = final_perf[sim]

            # Apply participation
            if perf > 1.0:
                ret = (perf - 1.0) * self.params.participation_rate
                if self.params.has_cap and self.params.cap_level:
                    ret = min(ret, self.params.cap_level - 1.0)
                payoffs[sim] = 1.0 + ret
            else:
                if self.params.has_floor and self.params.floor_level:
                    payoffs[sim] = max(self.params.floor_level, perf)
                else:
                    payoffs[sim] = perf

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': np.mean(final_perf < 1.0),
            'avg_coupons': 0,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_fixed_coupon(self):
        """Fixed coupon note - like a bond."""
        n_sims = self.params.n_simulations

        total_coupon = self.params.coupon_rate * self.params.maturity_years
        payoff = (1.0 + total_coupon)

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = np.full(n_sims, payoff * self.params.notional * disc)

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': 0,
            'avg_coupons': total_coupon,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_reverse_convertible(self):
        """Reverse convertible - high coupon, downside exposure."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]
        min_perf = np.min(performance, axis=1)

        total_coupon = self.params.coupon_rate * self.params.maturity_years
        payoffs = np.zeros(n_sims)
        ki_triggered = np.zeros(n_sims, dtype=bool)

        if self.params.has_knock_in and self.params.ki_barrier:
            ki_triggered = min_perf <= self.params.ki_barrier

        for sim in range(n_sims):
            if ki_triggered[sim] and final_perf[sim] < 1.0:
                payoffs[sim] = final_perf[sim] + total_coupon
            else:
                payoffs[sim] = 1.0 + total_coupon

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': np.mean(ki_triggered),
            'prob_loss': np.mean((ki_triggered) & (final_perf < 1.0)),
            'avg_coupons': total_coupon,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_range_accrual(self):
        """Range accrual - coupon accrues when in range."""
        n_sims = self.params.n_simulations
        n_steps = self.paths.shape[1] - 1
        performance = self._get_performance()

        lower = self.params.range_lower or 0.8
        upper = self.params.range_upper or 1.2

        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            days_in_range = np.sum((performance[sim, :] >= lower) & (performance[sim, :] <= upper))
            accrual_rate = days_in_range / (n_steps + 1)
            coupon = self.params.coupon_rate * self.params.maturity_years * accrual_rate
            payoffs[sim] = 1.0 + coupon

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': 0,
            'avg_coupons': np.mean(self.payoffs / self.params.notional / disc - 1.0),
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_bonus_certificate(self):
        """Bonus certificate - bonus if barrier not breached."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]
        min_perf = np.min(performance, axis=1)

        barrier = self.params.ki_barrier or 0.7
        bonus = self.params.cap_level or 1.2

        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            if min_perf[sim] > barrier:
                payoffs[sim] = max(bonus, final_perf[sim])
            else:
                payoffs[sim] = final_perf[sim]

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': np.mean(min_perf <= barrier),
            'prob_loss': np.mean(self.payoffs < self.params.notional * disc),
            'avg_coupons': 0,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def _payoff_discount_certificate(self):
        """Discount certificate - buy at discount, capped upside."""
        n_sims = self.params.n_simulations
        performance = self._get_performance()
        final_perf = performance[:, -1]

        cap = self.params.cap_level or 1.1

        payoffs = np.zeros(n_sims)

        for sim in range(n_sims):
            payoffs[sim] = min(final_perf[sim], cap)

        disc = np.exp(-self.params.risk_free_rate * self.params.maturity_years)
        self.payoffs = payoffs * self.params.notional * disc

        self.statistics = {
            'prob_autocall': 0,
            'prob_ki': 0,
            'prob_loss': np.mean(final_perf < 1.0),
            'avg_coupons': 0,
            'avg_autocall_period': 0,
            'n_observations': 1
        }

        return self.payoffs

    def price(self):
        """Run pricing."""
        np.random.seed(42)
        self._generate_paths()
        payoffs = self._calculate_payoffs()

        fair_value = np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(len(payoffs))

        return {
            'fair_value': fair_value,
            'std_error': std_error,
            'confidence_interval_95': (fair_value - 1.96 * std_error, fair_value + 1.96 * std_error),
            'prob_autocall': self.statistics.get('prob_autocall', 0),
            'prob_ki': self.statistics.get('prob_ki', 0),
            'prob_loss': self.statistics.get('prob_loss', 0),
            'expected_return': (fair_value - self.params.notional) / self.params.notional,
            'avg_coupons_received': self.statistics.get('avg_coupons', 0) * self.params.notional,
            'avg_autocall_period': self.statistics.get('avg_autocall_period', 0),
            'n_observations': self.statistics.get('n_observations', 0),
        }

    def calculate_greeks(self):
        """Calculate Greeks."""
        base = self.price()['fair_value']
        bump = 0.01

        # Delta
        orig = self.params.initial_levels.copy()
        self.params.initial_levels = orig * (1 + bump)
        self._generate_paths()
        up = self._calculate_payoffs().mean()
        self.params.initial_levels = orig * (1 - bump)
        self._generate_paths()
        down = self._calculate_payoffs().mean()
        delta = (up - down) / (2 * bump * orig[0])
        self.params.initial_levels = orig

        # Vega
        orig_vol = self.params.volatilities.copy()
        self.params.volatilities = orig_vol + 0.01
        self._generate_paths()
        up = self._calculate_payoffs().mean()
        self.params.volatilities = orig_vol - 0.01
        self._generate_paths()
        down = self._calculate_payoffs().mean()
        vega = (up - down) / 0.02
        self.params.volatilities = orig_vol

        # Theta
        orig_mat = self.params.maturity_years
        self.params.maturity_years = orig_mat - 1/365
        self._generate_paths()
        later = self._calculate_payoffs().mean()
        theta = (later - base) / (1/365)
        self.params.maturity_years = orig_mat

        # Rho
        orig_rate = self.params.risk_free_rate
        self.params.risk_free_rate = orig_rate + 0.0001
        self._generate_paths()
        up = self._calculate_payoffs().mean()
        self.params.risk_free_rate = orig_rate - 0.0001
        self._generate_paths()
        down = self._calculate_payoffs().mean()
        rho = (up - down) / 0.0002
        self.params.risk_free_rate = orig_rate

        self._generate_paths()
        return {'delta': delta, 'vega': vega, 'theta': theta, 'rho': rho}

    def calculate_risk_metrics(self):
        """Calculate risk metrics."""
        if self.payoffs is None:
            self.price()

        payoffs = self.payoffs
        var_95 = self.params.notional - np.percentile(payoffs, 5)
        var_99 = self.params.notional - np.percentile(payoffs, 1)
        es_95 = self.params.notional - np.mean(payoffs[payoffs <= np.percentile(payoffs, 5)])
        max_loss = self.params.notional - np.min(payoffs)

        return {
            'var_95': max(0, var_95),
            'var_99': max(0, var_99),
            'expected_shortfall_95': max(0, es_95),
            'max_loss': max(0, max_loss),
            'probability_of_loss': np.mean(payoffs < self.params.notional),
        }
