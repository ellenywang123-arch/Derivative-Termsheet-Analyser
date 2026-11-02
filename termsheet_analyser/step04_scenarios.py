"""04 — Construct educational scenario paths (not pricing simulations)."""
from __future__ import annotations

from datetime import datetime
from datetime import timedelta
import hashlib
import numpy as np
import re

def generate_constrained_path(start_val, end_val, n_points, vol=2.0, force_min=None, force_max=None, seed=None):
    """
    This function is now DEPRECATED for scenario visualization.
    Kept for backward compatibility.
    """
    if seed is not None:
        np.random.seed(seed)

    if start_val is None: start_val = 100.0
    if end_val is None: end_val = 100.0

    path = np.linspace(start_val, end_val, n_points)
    noise = np.random.normal(0, vol * 0.3, n_points)
    path = path + noise
    path[0] = start_val
    path[-1] = end_val

    return path

def generate_discrete_scenario_points(scenario_data, calendar, barrier_hierarchy, start_date, end_date):
    """
    Generate discrete price points at each observation date based on scenario logic.

    IMPORTANT:
    - Returns points for ALL observation dates (for x-axis display)
    - 'is_termination' marks where the product terminates
    - Points AFTER termination have 'price' = None (will not be plotted but date shown on x-axis)

    Returns: (points_list, total_obs_count, termination_obs_index)
    """
    points = []

    # Parse scenario information
    condition = str(scenario_data.get('condition', '')).lower()
    outcome = str(scenario_data.get('outcome', '')).lower()
    chart_type = str(scenario_data.get('chart_type', '')).lower()

    # Extract barrier levels
    ko_level = None
    ki_level = None
    coupon_level = None

    for b in barrier_hierarchy:
        b_type = str(b.get('type', '')).lower()
        try:
            val = float(re.sub(r'[^\d.\-]', '', str(b.get('value', ''))))
            if val < 5:
                val = val * 100
        except:
            continue

        if 'knock-out' in b_type or 'autocall' in b_type:
            ko_level = val
        elif 'knock-in' in b_type or 'protection' in b_type:
            ki_level = val
        elif 'coupon' in b_type:
            coupon_level = val

    # Default barriers if not found
    if ko_level is None:
        ko_level = 100.0
    if ki_level is None:
        ki_level = 60.0
    if coupon_level is None:
        coupon_level = ki_level if ki_level else 70.0

    # Collect ALL observation dates from calendar
    all_obs_dates = []
    for entry in calendar:
        try:
            d = datetime.strptime(entry.get('date', ''), '%Y-%m-%d')
            if start_date <= d <= end_date:
                levels = entry.get('levels', {})
                date_ko = ko_level
                for k, v in levels.items():
                    if 'knock-out' in k.lower() or 'autocall' in k.lower():
                        try:
                            date_ko = float(re.sub(r'[^\d.\-]', '', str(v)))
                            if date_ko < 5:
                                date_ko = date_ko * 100
                        except:
                            pass
                all_obs_dates.append({'date': d, 'ko_level': date_ko})
        except:
            continue

    # Sort ALL observations chronologically and assign sequential index
    all_obs_dates.sort(key=lambda x: x['date'])

    # Assign actual observation index (1-based)
    for idx, obs in enumerate(all_obs_dates):
        obs['obs_index'] = idx + 1

    total_obs_count = len(all_obs_dates)

    # If no observation dates found, create some
    if not all_obs_dates:
        total_days = (end_date - start_date).days
        for i in range(1, 32):  # 31 observations
            d = start_date + timedelta(days=int(total_days * i / 31))
            if d <= end_date:
                all_obs_dates.append({'date': d, 'ko_level': ko_level, 'obs_index': i})
        total_obs_count = len(all_obs_dates)

    # Add start point (not an observation, obs_index = 0)
    points.append({
        'date': start_date,
        'price': 100.0,
        'is_termination': False,
        'termination_reason': None,
        'obs_index': 0,
        'is_after_termination': False
    })

    # ============================================================
    # SCENARIO TYPE DETECTION - IMPROVED LOGIC
    # Priority order matters: check more specific conditions first
    # ============================================================

    # 1. Check for Autocall (Early Redemption)
    is_autocall = (
        'autocall' in chart_type or
        'autocall' in condition or
        ('early' in condition and 'redemption' in condition) or
        ('early' in condition and 'call' in condition)
    )

    # 2. Check for KI Recovery (V-shape rebound) - must check BEFORE ki_loss
    is_ki_recovery = (
        'ki_recovery' in chart_type or
        'recovery' in chart_type or
        'rebound' in condition or
        'recover' in condition or
        'v-shape' in condition or
        ('knock-in' in condition and 'recover' in condition)
    )

    # 3. Check for KI Loss - only if NOT recovery
    is_ki_loss = (
        not is_ki_recovery and
        (
            'ki_loss' in chart_type or
            'loss' in chart_type or
            ('knock-in' in condition and 'loss' in condition) or
            ('breach' in condition and 'loss' in outcome) or
            ('touched' in condition and 'loss' in outcome)
        )
    )

    # 4. Check for Safe Maturity - explicitly check for safe/par/no knock-in
    is_safe_maturity = (
        not is_autocall and
        not is_ki_loss and
        not is_ki_recovery and
        (
            'safe' in chart_type or
            'safe' in condition or
            'at par' in condition or
            'no knock-in' in condition or
            'above' in condition or
            ('maturity' in condition and 'loss' not in condition and 'knock-in' not in condition) or
            'full principal' in outcome or
            'principal protected' in outcome or
            '100%' in outcome
        )
    )

    # 5. Default: if none of above, check outcome for hints
    if not is_autocall and not is_ki_loss and not is_ki_recovery and not is_safe_maturity:
        # Check outcome for final determination
        if 'loss' in outcome or 'below' in outcome:
            is_ki_loss = True
        else:
            # Default to safe maturity
            is_safe_maturity = True

    np.random.seed(int.from_bytes(hashlib.sha256(condition.encode("utf-8")).digest()[:4], "big"))

    termination_obs_index = None

    # Determine termination point based on scenario type
    if is_autocall:
        # Find which observation triggers autocall
        month_match = re.search(r'month\s*(\d+)', condition)
        year_match = re.search(r'year\s*(\d+)', condition)

        if month_match:
            target_month = int(month_match.group(1))
            termination_obs_index = min(target_month, total_obs_count)
        elif year_match:
            target_year = int(year_match.group(1))
            termination_obs_index = min(target_year * 12, total_obs_count)
        else:
            termination_obs_index = max(1, total_obs_count // 3)
    else:
        # All other scenarios terminate at maturity (last observation)
        termination_obs_index = total_obs_count

    # ============================================================
    # GENERATE PRICES FOR ALL OBSERVATION DATES
    # ============================================================
    for i, obs in enumerate(all_obs_dates):
        actual_obs_idx = obs['obs_index']  # 1-based
        date_ko = obs['ko_level']
        is_after_termination = actual_obs_idx > termination_obs_index
        is_termination_point = actual_obs_idx == termination_obs_index

        # Calculate price based on scenario type
        if is_after_termination:
            price = None
            term_reason = None

        elif is_autocall:
            if actual_obs_idx < termination_obs_index:
                base_price = date_ko - np.random.uniform(5, 15)
                noise = np.random.uniform(-3, 3)
                price = max(ki_level + 5, base_price + noise)
                term_reason = None
            else:
                price = date_ko + np.random.uniform(2, 8)
                term_reason = f'Autocall Triggered (Obs {actual_obs_idx})'

        elif is_ki_loss:
            # Knock-in loss scenario
            drop_idx = total_obs_count // 3

            if actual_obs_idx < drop_idx:
                price = 100 - (actual_obs_idx / max(drop_idx, 1)) * (100 - ki_level - 5) + np.random.uniform(-3, 3)
                price = max(ki_level + 2, price)
            elif actual_obs_idx == drop_idx:
                price = ki_level - np.random.uniform(3, 10)
            else:
                progress = (actual_obs_idx - drop_idx) / max(total_obs_count - drop_idx, 1)
                end_target = scenario_data.get('end_price_target', ki_level - 20)
                try:
                    end_target = float(re.sub(r'[^\d.\-]', '', str(end_target)))
                    if end_target < 5:
                        end_target = end_target * 100
                except:
                    end_target = ki_level - 20

                price = ki_level - 5 + progress * (end_target - (ki_level - 5))
                price += np.random.uniform(-3, 3)

            term_reason = f'Maturity (Knock-in Loss, Obs {actual_obs_idx})' if is_termination_point else None

        elif is_ki_recovery:
            # V-shape recovery
            drop_idx = total_obs_count // 3
            recovery_start = total_obs_count // 2

            if actual_obs_idx < drop_idx:
                price = 100 - (actual_obs_idx / max(drop_idx, 1)) * (100 - ki_level + 5) + np.random.uniform(-2, 2)
            elif actual_obs_idx < recovery_start:
                price = ki_level - np.random.uniform(5, 15)
            else:
                progress = (actual_obs_idx - recovery_start) / max(total_obs_count - recovery_start, 1)
                price = (ki_level - 10) + progress * (105 - (ki_level - 10))
                price += np.random.uniform(-3, 3)

            if is_termination_point:
                price = max(price, 100 + np.random.uniform(2, 8))

            term_reason = f'Maturity (KI Recovery, Obs {actual_obs_idx})' if is_termination_point else None

        elif is_safe_maturity:
            # Safe maturity scenario: price stays above KI, ends at strike (100%)
            mid_point = (ki_level + date_ko) / 2
            amplitude = (date_ko - ki_level) / 4

            progress = actual_obs_idx / total_obs_count
            trend_target = 100.0

            base_price = mid_point * (1 - progress) + trend_target * progress
            noise = np.random.uniform(-amplitude * (1 - progress * 0.5), amplitude * (1 - progress * 0.5))
            price = base_price + noise

            price = max(ki_level + 3, min(date_ko - 3, price))

            # At maturity, redemption at strike (100%) - Safe/At Par
            if is_termination_point:
                price = 100.0

            # FIXED: Correct termination reason for safe maturity
            term_reason = f'Maturity (Safe/At Par, Obs {actual_obs_idx})' if is_termination_point else None

        else:
            # Fallback: treat as safe scenario
            price = 100 + np.random.uniform(-10, 10)
            price = max(ki_level + 3, min(date_ko - 3, price))
            if is_termination_point:
                price = 100.0
            term_reason = f'Maturity (At Par, Obs {actual_obs_idx})' if is_termination_point else None

        points.append({
            'date': obs['date'],
            'price': price,
            'is_termination': is_termination_point,
            'termination_reason': term_reason,
            'obs_index': actual_obs_idx,
            'is_after_termination': is_after_termination
        })

    return points, total_obs_count, termination_obs_index
