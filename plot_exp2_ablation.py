#!/usr/bin/env python3
"""
Plotting script for Experiment 2: Stochastic Forcing Ablation Study.

Generates publication-quality figures comparing phase trajectories (α̂, β̂) under
different ablation conditions (batch size, gradient clipping, Winsorization) against
the baseline.

Usage:
    python plot_exp2_ablation.py \
        --agg_dir <path_to_aggregated_dir> \
        --outdir <path_to_output_dir> \
        --dpi 300
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Set matplotlib style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


class Experiment2Plotter:
    """Plotter for Experiment 2 ablation study."""

    def __init__(self, agg_dir: str, outdir: str, dpi: int = 300,
                 models: List[str] = None, tau_cap: float = 1e6):
        """
        Initialize plotter.

        Args:
            agg_dir: Path to aggregated results directory
            outdir: Output directory for figures
            dpi: DPI for saved figures
            models: List of model names to plot
            tau_cap: Cap for tau values (for filtering outliers)
        """
        self.agg_dir = Path(agg_dir)
        self.outdir = Path(outdir)
        self.dpi = dpi
        self.tau_cap = tau_cap
        self.models = models or ['diag', 'gru', 'lstm']

        # Create output directory
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Define color palettes
        self.baseline_color = '#1f77b4'  # blue
        self.batch_colors = ['#ff7f0e', '#d62728', '#9467bd']  # orange, red, purple
        self.clip_colors = ['#2ca02c', '#17becf', '#bcbd22']  # green, cyan, yellow
        self.winsorize_colors = ['#e377c2', '#7f7f7f', '#8c564b']  # pink, gray, brown

        # Define line styles
        self.baseline_style = '-'
        self.batch_style = '--'
        self.clip_style = ':'
        self.winsorize_style = '-.'

        # Data storage
        self.data = {}  # model -> condition -> data dict

    @staticmethod
    def _time_axis(data: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
        """Use optimizer step when available, otherwise fall back to epoch."""
        if 'step' in data:
            step = np.asarray(data['step'], dtype=float)
            if np.any(np.isfinite(step)):
                return 'step', step
        return 'epoch', np.asarray(data['epoch'], dtype=float)

    def load_phase_trajectory(self, model: str, condition: str, value: str = None
                             ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load phase trajectory CSV for a given model/condition.

        Args:
            model: Model name (diag, gru, lstm)
            condition: Condition name (baseline, batch_ablation, etc.)
            value: Value parameter (batch size, clip threshold, etc.)

        Returns:
            Dictionary with loaded columns or None if file not found.
        """
        if value is not None:
            cond_dir = self.agg_dir / model / f"condition_{condition}_{value}"
        else:
            cond_dir = self.agg_dir / model / f"condition_{condition}"

        # Try aggregated file first, fall back to regular file
        for fname in ['phase_trajectory_aggregated.csv', 'phase_trajectory.csv']:
            fpath = cond_dir / fname
            if fpath.exists():
                try:
                    data = np.genfromtxt(fpath, delimiter=',', dtype=float,
                                        names=True, filling_values=np.nan)
                    loaded = {name: data[name] for name in data.dtype.names}
                    if fname == 'phase_trajectory_aggregated.csv':
                        normalized = {}
                        for name, values in loaded.items():
                            normalized[name] = values
                            if name.endswith('_mean'):
                                normalized[name[:-5]] = values
                            elif name.endswith('_se'):
                                normalized[name] = values
                        return normalized
                    return loaded
                except Exception as e:
                    print(f"Warning: Failed to load {fpath}: {e}")
                    return None

        return None

    def load_all_data(self, batch_values: List[str], clip_values: List[str],
                      winsorize_values: List[str]) -> None:
        """Load all phase trajectory data."""
        conditions = {
            'baseline': {'baseline': [None]},
            'batch_ablation': {f'batch_ablation_{v}': [v] for v in batch_values},
            'clip_ablation': {f'clip_ablation_{v}': [v] for v in clip_values},
            'winsorize_ablation': {f'winsorize_ablation_{v}': [v] for v in winsorize_values},
        }

        for model in self.models:
            self.data[model] = {}

            # Load baseline
            data = self.load_phase_trajectory(model, 'baseline')
            if data is not None:
                self.data[model]['baseline'] = data
                print(f"Loaded {model}/baseline")
            else:
                print(f"Warning: {model}/baseline not found")

            # Load batch ablations
            for batch_val in batch_values:
                cond_name = f'batch_ablation_{batch_val}'
                data = self.load_phase_trajectory(model, 'batch_ablation', batch_val)
                if data is not None:
                    self.data[model][cond_name] = data
                    print(f"Loaded {model}/{cond_name}")

            # Load clip ablations
            for clip_val in clip_values:
                cond_name = f'clip_ablation_{clip_val}'
                data = self.load_phase_trajectory(model, 'clip_ablation', clip_val)
                if data is not None:
                    self.data[model][cond_name] = data
                    print(f"Loaded {model}/{cond_name}")

            # Load winsorize ablations
            for winsorize_val in winsorize_values:
                cond_name = f'winsorize_ablation_{winsorize_val}'
                data = self.load_phase_trajectory(model, 'winsorize_ablation', winsorize_val)
                if data is not None:
                    self.data[model][cond_name] = data
                    print(f"Loaded {model}/{cond_name}")

    def plot_phase_trajectories_3panel(self) -> None:
        """
        Figure 1: Phase trajectories in (α̂, β̂) plane — one panel per model.

        Shows baseline and all ablations on separate subplots.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Phase Trajectories: Ablation Study', fontsize=14, fontweight='bold')

        for ax_idx, model in enumerate(self.models):
            ax = axes[ax_idx]

            if model not in self.data or len(self.data[model]) == 0:
                ax.text(0.5, 0.5, f'No data for {model}', ha='center', va='center',
                       transform=ax.transAxes)
                continue

            # Plot baseline
            if 'baseline' in self.data[model]:
                data = self.data[model]['baseline']
                ax.plot(data['alpha_hat'], data['beta_hat'],
                       color=self.baseline_color, linestyle=self.baseline_style,
                       linewidth=2, label='Baseline', marker='o', markersize=4,
                       markevery=max(1, len(data['alpha_hat'])//10))

            # Plot batch ablations
            for i, batch_val in enumerate(['2048', '4096', '8192']):
                cond_name = f'batch_ablation_{batch_val}'
                if cond_name in self.data[model]:
                    data = self.data[model][cond_name]
                    ax.plot(data['alpha_hat'], data['beta_hat'],
                           color=self.batch_colors[i], linestyle=self.batch_style,
                           linewidth=1.5, alpha=0.7,
                           label=f'Batch {batch_val}', marker='s', markersize=3,
                           markevery=max(1, len(data['alpha_hat'])//10))

            # Plot clip ablations
            for i, clip_val in enumerate(['0.1', '0.01', '0.001']):
                cond_name = f'clip_ablation_{clip_val}'
                if cond_name in self.data[model]:
                    data = self.data[model][cond_name]
                    ax.plot(data['alpha_hat'], data['beta_hat'],
                           color=self.clip_colors[i], linestyle=self.clip_style,
                           linewidth=1.5, alpha=0.7,
                           label=f'Clip {clip_val}', marker='^', markersize=3,
                           markevery=max(1, len(data['alpha_hat'])//10))

            # Plot winsorize ablations
            for i, wins_val in enumerate(['95', '90', '80']):
                cond_name = f'winsorize_ablation_{wins_val}'
                if cond_name in self.data[model]:
                    data = self.data[model][cond_name]
                    ax.plot(data['alpha_hat'], data['beta_hat'],
                           color=self.winsorize_colors[i], linestyle=self.winsorize_style,
                           linewidth=1.5, alpha=0.7,
                           label=f'Wins {wins_val}', marker='D', markersize=3,
                           markevery=max(1, len(data['alpha_hat'])//10))

            # Add critical lines
            ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='β=1 (critical)')
            ax.axvline(x=2, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='α=2 (Gaussian)')

            ax.set_xlabel('α̂ (gradient tail index)', fontsize=12)
            ax.set_ylabel('β̂ (spectral exponent)', fontsize=12)
            ax.set_title(model.upper(), fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, 'figure1_phase_trajectories')

    def plot_beta_timeseries_by_ablation(self) -> None:
        """
        Figure 2: β̂ time series — one panel per ablation type.

        All models on the same panel, organized by ablation type.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('β̂ Time Series: Ablation Conditions', fontsize=14, fontweight='bold')

        ablation_types = [
            ('batch_ablation', ['2048', '4096', '8192'], self.batch_colors, self.batch_style),
            ('clip_ablation', ['0.1', '0.01', '0.001'], self.clip_colors, self.clip_style),
            ('winsorize_ablation', ['95', '90', '80'], self.winsorize_colors, self.winsorize_style),
        ]

        for ax_idx, (ablation_type, values, colors, line_style) in enumerate(ablation_types):
            ax = axes[ax_idx]

            for model_idx, model in enumerate(self.models):
                if model not in self.data:
                    continue

                # Plot baseline for this model
                if 'baseline' in self.data[model]:
                    data = self.data[model]['baseline']
                    _, time_axis = self._time_axis(data)
                    ax.plot(time_axis, data['beta_hat'],
                           color=f'C{model_idx}', linestyle=self.baseline_style,
                           linewidth=2, alpha=0.8, label=f'{model.upper()} (baseline)',
                           marker='o', markersize=3, markevery=max(1, len(time_axis)//8))

                # Plot ablation conditions
                for val_idx, val in enumerate(values):
                    cond_name = f'{ablation_type}_{val}'
                    if cond_name in self.data[model]:
                        data = self.data[model][cond_name]
                        _, time_axis = self._time_axis(data)
                        # Opacity proportional to ablation strength (inverse of value)
                        if ablation_type == 'batch_ablation':
                            alpha = 0.4 + 0.3 * (val_idx / len(values))
                        elif ablation_type == 'clip_ablation':
                            alpha = 0.4 + 0.3 * ((len(values) - val_idx) / len(values))
                        else:  # winsorize
                            alpha = 0.4 + 0.3 * ((len(values) - val_idx) / len(values))

                        ax.plot(time_axis, data['beta_hat'],
                               color=colors[val_idx], linestyle=line_style,
                               linewidth=1.5, alpha=alpha,
                               label=f'{model.upper()} {ablation_type.split("_")[0]} {val}',
                               marker='s', markersize=2, markevery=max(1, len(time_axis)//8))

            # Add critical line
            ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='β=1')

            ax.set_xlabel('Optimizer Step', fontsize=12)
            ax.set_ylabel('β̂ (spectral exponent)', fontsize=12)
            ablation_title = ablation_type.replace('_', ' ').title()
            ax.set_title(f'{ablation_title}', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, 'figure2_beta_timeseries')

    def plot_alpha_timeseries_by_ablation(self) -> None:
        """
        Figure 3: α̂ time series — same layout as Figure 2.

        Shows how gradient noise statistics respond to each intervention.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('α̂ Time Series: Ablation Conditions', fontsize=14, fontweight='bold')

        ablation_types = [
            ('batch_ablation', ['2048', '4096', '8192'], self.batch_colors, self.batch_style),
            ('clip_ablation', ['0.1', '0.01', '0.001'], self.clip_colors, self.clip_style),
            ('winsorize_ablation', ['95', '90', '80'], self.winsorize_colors, self.winsorize_style),
        ]

        for ax_idx, (ablation_type, values, colors, line_style) in enumerate(ablation_types):
            ax = axes[ax_idx]

            for model_idx, model in enumerate(self.models):
                if model not in self.data:
                    continue

                # Plot baseline for this model
                if 'baseline' in self.data[model]:
                    data = self.data[model]['baseline']
                    _, time_axis = self._time_axis(data)
                    ax.plot(time_axis, data['alpha_hat'],
                           color=f'C{model_idx}', linestyle=self.baseline_style,
                           linewidth=2, alpha=0.8, label=f'{model.upper()} (baseline)',
                           marker='o', markersize=3, markevery=max(1, len(time_axis)//8))

                # Plot ablation conditions
                for val_idx, val in enumerate(values):
                    cond_name = f'{ablation_type}_{val}'
                    if cond_name in self.data[model]:
                        data = self.data[model][cond_name]
                        _, time_axis = self._time_axis(data)
                        # Opacity proportional to ablation strength
                        if ablation_type == 'batch_ablation':
                            alpha = 0.4 + 0.3 * (val_idx / len(values))
                        elif ablation_type == 'clip_ablation':
                            alpha = 0.4 + 0.3 * ((len(values) - val_idx) / len(values))
                        else:  # winsorize
                            alpha = 0.4 + 0.3 * ((len(values) - val_idx) / len(values))

                        ax.plot(time_axis, data['alpha_hat'],
                               color=colors[val_idx], linestyle=line_style,
                               linewidth=1.5, alpha=alpha,
                               label=f'{model.upper()} {ablation_type.split("_")[0]} {val}',
                               marker='s', markersize=2, markevery=max(1, len(time_axis)//8))

            # Add critical line
            ax.axvline(x=2, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='α=2')

            ax.set_xlabel('Optimizer Step', fontsize=12)
            ax.set_ylabel('α̂ (gradient tail index)', fontsize=12)
            ablation_title = ablation_type.replace('_', ' ').title()
            ax.set_title(f'{ablation_title}', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, 'figure3_alpha_timeseries')

    def plot_final_beta_summary(self) -> None:
        """
        Figure 4: Summary bar chart of final-epoch β̂ for each (model, condition).

        Grouped by model, bars colored by condition.
        """
        # Collect final β̂ values for each (model, condition)
        final_betas = defaultdict(dict)
        final_betas_se = defaultdict(dict)

        for model in self.models:
            if model not in self.data:
                continue

            for cond_name, data in self.data[model].items():
                if 'beta_hat' in data and len(data['beta_hat']) > 0:
                    # Use last non-NaN value
                    valid_idx = ~np.isnan(data['beta_hat'])
                    if np.any(valid_idx):
                        final_betas[model][cond_name] = data['beta_hat'][valid_idx][-1]

                        # Try to get SE if available
                        if 'beta_hat_se' in data:
                            se_valid_idx = ~np.isnan(data['beta_hat_se'])
                            if np.any(se_valid_idx):
                                final_betas_se[model][cond_name] = data['beta_hat_se'][se_valid_idx][-1]

        if not final_betas:
            print("Warning: No final β̂ values to plot")
            return

        # Prepare data for grouped bar chart
        all_conditions = set()
        for model_conds in final_betas.values():
            all_conditions.update(model_conds.keys())
        all_conditions = sorted(list(all_conditions))

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set up bar positions
        n_models = len(self.models)
        n_conditions = len(all_conditions)
        bar_width = 0.8 / n_models
        x_positions = np.arange(n_conditions)

        # Color map for conditions
        color_map = self._get_condition_colors()

        # Plot bars
        for model_idx, model in enumerate(self.models):
            betas = []
            errors = []
            for cond in all_conditions:
                if cond in final_betas.get(model, {}):
                    betas.append(final_betas[model][cond])
                    errors.append(final_betas_se.get(model, {}).get(cond, 0))
                else:
                    betas.append(0)
                    errors.append(0)

            offset = (model_idx - n_models / 2 + 0.5) * bar_width
            ax.bar(x_positions + offset, betas, bar_width, label=model.upper(),
                  error_kw={'elinewidth': 1, 'capsize': 3}, yerr=errors, alpha=0.8)

        # Add critical line
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.6, label='β=1 (critical)')

        ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final β̂ (spectral exponent)', fontsize=12, fontweight='bold')
        ax.set_title('Final β̂ Across Ablation Conditions', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([cond.replace('_', '\n') for cond in all_conditions],
                           fontsize=10, rotation=45, ha='right')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_figure(fig, 'figure4_final_beta_summary')

    def _get_condition_colors(self) -> Dict[str, str]:
        """Map condition names to colors."""
        color_map = {}
        color_map['baseline'] = self.baseline_color

        for i, val in enumerate(['2048', '4096', '8192']):
            color_map[f'batch_ablation_{val}'] = self.batch_colors[i]

        for i, val in enumerate(['0.1', '0.01', '0.001']):
            color_map[f'clip_ablation_{val}'] = self.clip_colors[i]

        for i, val in enumerate(['95', '90', '80']):
            color_map[f'winsorize_ablation_{val}'] = self.winsorize_colors[i]

        return color_map

    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure in both PDF and PNG formats."""
        pdf_path = self.outdir / f'{name}.pdf'
        png_path = self.outdir / f'{name}.png'

        fig.savefig(pdf_path, dpi=self.dpi, bbox_inches='tight', format='pdf')
        fig.savefig(png_path, dpi=self.dpi, bbox_inches='tight', format='png')

        print(f"Saved {pdf_path}")
        print(f"Saved {png_path}")

        plt.close(fig)

    def run(self, batch_values: List[str], clip_values: List[str],
            winsorize_values: List[str]) -> None:
        """Generate all figures."""
        print("Loading data...")
        self.load_all_data(batch_values, clip_values, winsorize_values)

        print("\nGenerating Figure 1: Phase trajectories...")
        self.plot_phase_trajectories_3panel()

        print("Generating Figure 2: β̂ time series...")
        self.plot_beta_timeseries_by_ablation()

        print("Generating Figure 3: α̂ time series...")
        self.plot_alpha_timeseries_by_ablation()

        print("Generating Figure 4: Final β̂ summary...")
        self.plot_final_beta_summary()

        print(f"\nAll figures saved to {self.outdir}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Experiment 2 (Stochastic Forcing Ablation) results')
    parser.add_argument('--agg_dir', required=True,
                       help='Path to aggregated results directory')
    parser.add_argument('--outdir', required=True,
                       help='Output directory for figures')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures')
    parser.add_argument('--models', default='diag,gru,lstm',
                       help='Comma-separated list of models to plot')
    parser.add_argument('--batch_values', default='2048,4096,8192',
                       help='Comma-separated batch sizes')
    parser.add_argument('--clip_values', default='0.1,0.01,0.001',
                       help='Comma-separated clip thresholds')
    parser.add_argument('--winsorize_values', default='95,90,80',
                       help='Comma-separated Winsorize percentiles')
    parser.add_argument('--tau_cap', type=float, default=1e6,
                       help='Cap for tau values')

    args = parser.parse_args()

    # Parse list arguments
    models = args.models.split(',')
    batch_values = args.batch_values.split(',')
    clip_values = args.clip_values.split(',')
    winsorize_values = args.winsorize_values.split(',')

    # Check that agg_dir exists
    if not Path(args.agg_dir).exists():
        print(f"Error: --agg_dir '{args.agg_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Create plotter and run
    plotter = Experiment2Plotter(
        agg_dir=args.agg_dir,
        outdir=args.outdir,
        dpi=args.dpi,
        models=models,
        tau_cap=args.tau_cap
    )

    plotter.run(batch_values, clip_values, winsorize_values)


if __name__ == '__main__':
    main()
