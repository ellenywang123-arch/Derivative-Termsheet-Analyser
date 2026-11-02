"""05 — Render product lifecycle and scenario charts."""
from __future__ import annotations

from .step04_scenarios import generate_discrete_scenario_points
from datetime import datetime
from datetime import timedelta
from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import textwrap

class LifecycleFlowVisualizer:
    """
    通用生命周期逻辑流可视化 (基于 Graphviz 动态渲染)
    """

    def __init__(self):
        self.styles = {
            'graph': {
                'rankdir': 'TB',
                'splines': 'ortho',
                'nodesep': '0.6',
                'ranksep': '0.6',
                'fontname': 'Helvetica',
            },
            'types': {
                'start': {
                    'shape': 'box', 'style': 'filled,rounded',
                    'fillcolor': '#e1f5fe', 'color': '#01579b',
                    'fontname': 'Helvetica-Bold', 'fontsize': '10'
                },
                'action': {
                    'shape': 'note',
                    'style': 'filled',
                    'fillcolor': '#fff9c4', 'color': '#fbc02d',
                    'fontname': 'Helvetica', 'fontsize': '10'
                },
                'decision': {
                    'shape': 'diamond', 'style': 'filled',
                    'fillcolor': '#fff3e0', 'color': '#e65100',
                    'fontname': 'Helvetica', 'fontsize': '9'
                },
                'exit': {
                    'shape': 'doubleoctagon', 'style': 'filled',
                    'fillcolor': '#e8f5e9', 'color': '#2e7d32',
                    'fontname': 'Helvetica-Bold', 'fontsize': '10'
                },
                'default': {
                    'shape': 'box', 'style': 'filled',
                    'fillcolor': '#f5f5f5', 'color': '#9e9e9e'
                }
            },
            'edge': {
                'fontname': 'Helvetica',
                'fontsize': '8',
                'color': '#424242',
                'arrowsize': '0.7'
            }
        }

    def plot_logic_flow(self, viz_data, save_path=None):
        dot = Digraph(comment='Dynamic Lifecycle Flow')
        dot.attr(**self.styles['graph'])
        dot.edge_attr.update(self.styles['edge'])

        flow_design = viz_data.get('flowchart_design', {})
        if not flow_design:
            return None

        nodes = flow_design.get('nodes', [])
        edges = flow_design.get('edges', [])
        existing_node_ids = set()

        for node in nodes:
            n_id = node.get('id')
            n_label = node.get('label', 'Node')
            n_type = node.get('type', 'default').lower()

            if not n_id: continue

            style_attrs = self.styles['types'].get(n_type, self.styles['types']['default']).copy()
            clean_label = str(n_label).replace('\\n', '\n')

            dot.node(n_id, label=clean_label, **style_attrs)
            existing_node_ids.add(n_id)

        for edge in edges:
            src = edge.get('from')
            dst = edge.get('to')
            label = edge.get('label', '')
            style = edge.get('style', 'solid')

            if src in existing_node_ids and dst in existing_node_ids:
                attrs = {}
                if label: attrs['xlabel'] = label
                if style == 'dashed':
                    attrs['style'] = 'dashed'
                    attrs['constraint'] = 'false'

                dot.edge(src, dst, **attrs)

        if save_path is None:
            rnd_id = random.randint(10000, 99999)
            save_path = f"dynamic_flow_{rnd_id}"

        base_path = os.path.splitext(save_path)[0]

        try:
            output_path = dot.render(filename=base_path, format='png', cleanup=True, view=False)
            return output_path
        except Exception as e:
            print(f"Graphviz rendering failed: {e}")
            return None

class EducationalVisualizer:
    """Visualizes scenarios with discrete points at observation dates."""

    BARRIER_CONFIG = {
        'Knock-out': {'color': '#228B22', 'style': '--', 'label': 'Knock-out / Autocall', 'zorder': 4, 'lw': 2.0},
        'Knock-in': {'color': '#DC143C', 'style': '-', 'label': 'Knock-in / Protection', 'zorder': 2, 'lw': 2.0},
        'Coupon': {'color': '#4169E1', 'style': '-.', 'label': 'Coupon Barrier', 'zorder': 3, 'lw': 1.5},
        'Strike': {'color': '#696969', 'style': ':', 'label': 'Strike (100%)', 'zorder': 5, 'lw': 1.2}
    }

    # Standardized Y-axis range
    Y_AXIS_MIN = 0
    Y_AXIS_MAX = 110

    def _normalize_value(self, val: float) -> float:
        if val is None:
            return None
        try:
            val = float(val)
        except:
            return None
        if np.isnan(val):
            return None
        if val == 0:
            return 0.0
        if abs(val) < 5.0:
            return val * 100.0
        return val

    def _extract_barrier_levels(self, barrier_hierarchy):
        """Extract and normalize barrier levels from hierarchy."""
        levels = {'Knock-out': None, 'Knock-in': None, 'Coupon': None}

        for b in barrier_hierarchy:
            b_type = str(b.get('type', '')).lower()
            try:
                val = float(re.sub(r'[^\d.\-]', '', str(b.get('value', ''))))
                val = self._normalize_value(val)
            except:
                continue

            if 'knock-out' in b_type or 'autocall' in b_type:
                levels['Knock-out'] = val
            elif 'knock-in' in b_type or 'protection' in b_type:
                levels['Knock-in'] = val
            elif 'coupon' in b_type:
                levels['Coupon'] = val

        return levels

    def _draw_step_barrier(self, ax, x_min, x_max, schedule, initial_level, style):
        """
        Draw a step/ladder barrier line that starts horizontally from y-axis.
        Does NOT draw vertical line from 0 - starts directly at the barrier level.

        Args:
            ax: matplotlib axis
            x_min: start date
            x_max: end date
            schedule: list of {'date': datetime, 'level': float}
            initial_level: initial barrier level
            style: dict with 'color', 'style', 'lw', 'label', 'zorder'
        """
        if not schedule and initial_level is None:
            return

        # Build list of (date, level) changes
        changes = []

        # Start with initial level at x_min
        current_level = initial_level if initial_level else (schedule[0]['level'] if schedule else 100.0)
        changes.append({'date': x_min, 'level': current_level})

        # Add all schedule changes
        for entry in schedule:
            if entry['date'] > x_min and entry['date'] <= x_max:
                changes.append(entry)

        # Sort by date
        changes.sort(key=lambda x: x['date'])

        # Draw horizontal segments and vertical drops (but NOT from 0)
        first_segment = True
        for i, change in enumerate(changes):
            current_date = change['date']
            current_level = change['level']

            # Determine end of this horizontal segment
            if i + 1 < len(changes):
                next_date = changes[i + 1]['date']
            else:
                next_date = x_max

            # Draw horizontal line at current level
            ax.hlines(y=current_level, xmin=current_date, xmax=next_date,
                     color=style['color'], ls=style['style'], lw=style['lw'],
                     label=style['label'] if first_segment else None,
                     zorder=style['zorder'])
            first_segment = False

            # Draw vertical drop to next level (if there is a next level)
            if i + 1 < len(changes):
                next_level = changes[i + 1]['level']
                ax.vlines(x=next_date, ymin=min(current_level, next_level), ymax=max(current_level, next_level),
                         color=style['color'], ls=style['style'], lw=style['lw'],
                         zorder=style['zorder'])

    def plot_professional_scenario(self, scenario_data, meta_data, index=1, output_directory=None):
        plt.close('all')

        # Parse dates
        dates_info = meta_data.get('dates', {})
        try:
            start_dt = datetime.strptime(dates_info.get('start', '2024-01-01'), '%Y-%m-%d')
            end_dt = datetime.strptime(dates_info.get('end', '2025-01-01'), '%Y-%m-%d')
        except:
            start_dt = datetime.now()
            end_dt = start_dt + timedelta(days=365)

        # Get calendar and barriers
        calendar = meta_data.get('calendar', [])
        barrier_hierarchy = meta_data.get('barrier_hierarchy', [])
        barrier_levels = self._extract_barrier_levels(barrier_hierarchy)

        # Generate discrete scenario points for ALL observations
        scenario_points, total_obs_count, termination_obs_index = generate_discrete_scenario_points(
            scenario_data, calendar, barrier_hierarchy, start_dt, end_dt
        )

        # Separate points: before/at termination vs after termination
        points_with_price = [pt for pt in scenario_points if pt['price'] is not None]
        all_obs_points = [pt for pt in scenario_points if pt['obs_index'] > 0]

        # Extract data for plotting (only points with prices)
        plot_dates = [pt['date'] for pt in points_with_price]
        plot_prices = [pt['price'] for pt in points_with_price]
        plot_obs_indices = [pt['obs_index'] for pt in points_with_price]

        # Find termination point
        termination_point = None
        for pt in scenario_points:
            if pt.get('is_termination') and pt['price'] is not None:
                termination_point = pt
                break

        # Create figure
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.25,
                              left=0.06, right=0.94, top=0.88, bottom=0.15)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # X-AXIS RANGE: ALWAYS FROM START TO END (same for all scenarios)
        x_min = start_dt
        x_max = end_dt

        # ============================================================
        # PLOT KNOCK-OUT BARRIER (GREEN) - LADDER/STEP SHAPE
        # Using custom method to avoid vertical line from 0
        # ============================================================
        ko_schedule = []
        for entry in calendar:
            try:
                d = datetime.strptime(entry.get('date', ''), '%Y-%m-%d')
                levels = entry.get('levels', {})
                for k, v in levels.items():
                    if 'knock-out' in k.lower() or 'autocall' in k.lower():
                        try:
                            val = float(re.sub(r'[^\d.\-]', '', str(v)))
                            val = self._normalize_value(val)
                            if val is not None:
                                ko_schedule.append({'date': d, 'level': val})
                        except:
                            pass
            except:
                continue

        ko_schedule.sort(key=lambda x: x['date'])

        initial_ko = barrier_levels.get('Knock-out')
        if not initial_ko and ko_schedule:
            initial_ko = ko_schedule[0]['level']
        if not initial_ko:
            initial_ko = 100.0

        # Use custom step drawing method
        self._draw_step_barrier(
            ax1, x_min, x_max, ko_schedule, initial_ko,
            self.BARRIER_CONFIG['Knock-out']
        )

        # ============================================================
        # PLOT KNOCK-IN BARRIER (RED) - HORIZONTAL LINE
        # ============================================================
        ki_val = barrier_levels.get('Knock-in')
        if ki_val is not None and ki_val > 0:
            style = self.BARRIER_CONFIG['Knock-in']
            ax1.hlines(y=ki_val, xmin=x_min, xmax=x_max,
                      color=style['color'], ls=style['style'], lw=style['lw'],
                      label=style['label'], zorder=style['zorder'])
            ax1.fill_between([x_min, x_max], 0, ki_val,
                           color='red', alpha=0.05, zorder=1)

        # ============================================================
        # PLOT COUPON BARRIER (BLUE)
        # ============================================================
        cpn_val = barrier_levels.get('Coupon')
        if cpn_val is not None and cpn_val > 0 and cpn_val != ki_val:
            style = self.BARRIER_CONFIG['Coupon']
            ax1.hlines(y=cpn_val, xmin=x_min, xmax=x_max,
                      color=style['color'], ls=style['style'], lw=style['lw'],
                      label=style['label'], zorder=style['zorder'])

        # ============================================================
        # PLOT STRIKE LINE (100%)
        # ============================================================
        ax1.hlines(y=100, xmin=x_min, xmax=x_max,
                  color=self.BARRIER_CONFIG['Strike']['color'],
                  ls=self.BARRIER_CONFIG['Strike']['style'],
                  lw=self.BARRIER_CONFIG['Strike']['lw'],
                  label='Strike (100%)', zorder=5, alpha=0.7)

        # ============================================================
        # PLOT ASSET PRICE (Points + Line) - ONLY UP TO TERMINATION
        # ============================================================
        if len(plot_dates) > 1:
            ax1.plot(plot_dates, plot_prices, color='#1a1a1a', lw=1.5, zorder=9, alpha=0.8)

        # Plot regular observation points (excluding start and termination)
        regular_points_dates = []
        regular_points_prices = []
        for pt in points_with_price:
            if pt['obs_index'] > 0 and not pt.get('is_termination'):
                regular_points_dates.append(pt['date'])
                regular_points_prices.append(pt['price'])

        if regular_points_dates:
            ax1.scatter(regular_points_dates, regular_points_prices,
                       color='#1a1a1a', s=35, zorder=10,
                       edgecolors='white', linewidths=0.5,
                       label='Asset Price')

        # ============================================================
        # TERMINATION HANDLING
        # ============================================================
        if termination_point:
            term_date = termination_point['date']
            term_price = termination_point['price']
            term_reason = termination_point.get('termination_reason', 'Termination')

            # Draw vertical termination line
            ax1.axvline(x=term_date, color='#FF6B00', ls='-', lw=2.5, zorder=15, alpha=0.9)

            # Mark termination point
            ax1.scatter([term_date], [term_price],
                       color='#FF6B00', s=150, zorder=20,
                       marker='X', edgecolors='white', linewidths=2,
                       label='Termination Point')

            # Add termination label
            label_y = self.Y_AXIS_MAX - 5

            ax1.annotate(term_reason,
                        xy=(term_date, label_y),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='#FF6B00',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#FF6B00', lw=1.5),
                        ha='left', va='top', zorder=25)

            # Annotate final price
            ax1.annotate(f"{term_price:.1f}%",
                        xy=(term_date, term_price),
                        xytext=(-35, -15), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#1a1a1a', alpha=0.9),
                        arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1),
                        zorder=21)

        # ============================================================
        # X-AXIS: SHOW ALL OBSERVATION DATES
        # ============================================================
        tick_dates = []
        tick_labels = []

        # Add start
        tick_dates.append(start_dt)
        tick_labels.append(f"{start_dt.strftime('%m/%d/%y')}\n(Start)")

        # Add ALL observation points (limit display to avoid crowding)
        if len(all_obs_points) <= 12:
            display_obs = all_obs_points
        else:
            indices = list(np.linspace(0, len(all_obs_points)-1, 10, dtype=int))

            if termination_obs_index and termination_obs_index <= len(all_obs_points):
                term_list_idx = termination_obs_index - 1
                if term_list_idx not in indices:
                    indices.append(term_list_idx)
                    indices = sorted(set(indices))

            display_obs = [all_obs_points[i] for i in indices if i < len(all_obs_points)]

        for pt in display_obs:
            actual_idx = pt['obs_index']
            d = pt['date']
            is_term = pt.get('is_termination', False)
            is_after = pt.get('is_after_termination', False)

            tick_dates.append(d)

            if is_term:
                term_type = pt.get('termination_reason', 'End')
                if 'Autocall' in str(term_type):
                    short_reason = 'Autocall'
                elif 'KI Loss' in str(term_type):
                    short_reason = 'KI Loss'
                elif 'KI Recovery' in str(term_type):
                    short_reason = 'Recovery'
                elif 'Safe' in str(term_type):
                    short_reason = 'Safe'
                else:
                    short_reason = 'End'
                tick_labels.append(f"{d.strftime('%m/%d/%y')}\n★ {short_reason}\n(Obs {actual_idx})")
            elif is_after:
                tick_labels.append(f"{d.strftime('%m/%d/%y')}\n(Obs {actual_idx})")
            else:
                tick_labels.append(f"{d.strftime('%m/%d/%y')}\n(Obs {actual_idx})")

        ax1.set_xticks(tick_dates)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)

        # Gray out labels after termination
        if termination_obs_index:
            for i, label in enumerate(ax1.get_xticklabels()):
                if i > 0 and i <= len(display_obs):
                    pt = display_obs[i-1] if i-1 < len(display_obs) else None
                    if pt and pt.get('is_after_termination'):
                        label.set_color('gray')
                        label.set_alpha(0.5)

        # Set x-axis limits to FULL range
        ax1.set_xlim(x_min, x_max)

        # ============================================================
        # Y-AXIS: STANDARDIZED RANGE (0 to 110)
        # ============================================================
        ax1.set_ylim(self.Y_AXIS_MIN, self.Y_AXIS_MAX)

        # ============================================================
        # LEGEND
        # ============================================================
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(),
                   loc='lower center', bbox_to_anchor=(0.5, 1.02),
                   ncol=5, frameon=False, fontsize=8)

        # ============================================================
        # TITLE
        # ============================================================
        raw_title = f"SCENARIO {index}: {scenario_data.get('condition', '')}"
        wrapped_title = "\n".join(textwrap.wrap(raw_title, width=80))
        fig.suptitle(wrapped_title, fontsize=12, fontweight='bold', y=0.98)

        ax1.set_ylabel("Price Level (%)", fontsize=10)
        ax1.grid(True, ls=':', alpha=0.4, zorder=0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # ============================================================
        # ANALYSIS BOX
        # ============================================================
        ax2.axis('off')

        term_obs_display = termination_point.get('obs_index', 'N/A') if termination_point else 'N/A'
        term_date_str = termination_point['date'].strftime('%Y-%m-%d') if termination_point else 'N/A'
        final_price_str = f"{plot_prices[-1]:.1f}%" if plot_prices else "N/A"

        analysis_content = (
            f"SCENARIO ANALYSIS:\n"
            f"• Condition: {scenario_data.get('condition')}\n"
            f"• Outcome: {scenario_data.get('outcome', 'N/A')}\n"
            f"• Termination: {termination_point.get('termination_reason', 'N/A') if termination_point else 'N/A'}\n"
            f"• Termination Date: {term_date_str} (Observation {term_obs_display} of {total_obs_count})\n"
            f"• Final Price: {final_price_str}"
        )

        wrapped_lines = []
        for line in analysis_content.split('\n'):
            wrapped_lines.append("\n".join(textwrap.wrap(line, width=110)))
        wrapped_analysis = "\n".join(wrapped_lines)

        ax2.text(0.02, 0.95, wrapped_analysis, fontsize=9, va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.5', fc='#FAFAFA', ec='#DDDDDD', lw=1),
                 transform=ax2.transAxes, family='monospace')

        fname = f"scenario_{index}_{np.random.randint(10000, 99999)}.png"
        if output_directory is not None:
            os.makedirs(output_directory, exist_ok=True)
            fname = os.path.join(output_directory, fname)
        plt.savefig(fname, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close()
        return fname
