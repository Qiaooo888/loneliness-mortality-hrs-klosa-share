# ============================================================
# Part 4: Model Evaluation — T019 Loneliness → Mortality (IPTW)
# Weighted Cox regression via R survival::coxph
# Per-database analysis + meta-analysis + survival curves
# Generated: 2026-05-01
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
from pathlib import Path
import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OTHER_DIR = OUTPUT_DIR / "intermediate"
TABLE_DIR = PROJECT_ROOT / "tables"
FIG_DIR = PROJECT_ROOT / "figures"
SUPP_DIR = PROJECT_ROOT / "Supplement"

for d in [FIG_DIR, TABLE_DIR, SUPP_DIR]:
    os.makedirs(d, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "Part4_result.txt"
R_SCRIPT = Path(__file__).parent / "part4_cox_analysis.R"
R_INPUT = OTHER_DIR / "cox_input.csv"
R_OUTPUT = OTHER_DIR / "cox_results.csv"


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    with open(RESULT_FILE, 'w', encoding='utf-8') as rf:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.__stdout__, rf)
        try:
            print("=" * 70)
            print("Part 4: Model Evaluation — T019 IPTW (Weighted Cox)")
            print("=" * 70)

            # ============================================================
            # Step 4.1: Load data
            # ============================================================
            print("\n[Step 4.1] Loading data...")
            df = pd.read_csv(OTHER_DIR / "cleanDataPart3.csv")
            mapping_df = pd.read_csv(DATA_DIR / "variable_name_mapping.csv")
            name_map = dict(zip(mapping_df['database_name'], mapping_df['sci_name']))
            print(f"  N={len(df):,}")
            print(f"  Events={df['event'].sum():,} ({df['event'].mean()*100:.1f}%)")
            print(f"  Columns: {[c for c in df.columns if 'iptw' in c or 'ps_' in c]}")

            # ============================================================
            # Step 4.2: Prepare data for R
            # ============================================================
            print("\n[Step 4.2] Preparing data for R Cox regression...")
            r_cols = ['database', 'loneliness', 'event', 'time_years', 'iptw']
            for db in ['HRS', 'KLoSA', 'SHARE']:
                if f'iptw_{db}' in df.columns:
                    r_cols.append(f'iptw_{db}')

            r_df = df[r_cols].copy()
            r_df.to_csv(R_INPUT, index=False)
            print(f"  Saved cox_input.csv ({len(r_df):,} rows, {len(r_df.columns)} cols)")

            # ============================================================
            # Step 4.3: Run R weighted Cox regression
            # ============================================================
            print("\n[Step 4.3] Running R weighted Cox regression...")
            print(f"  R script: {R_SCRIPT}")

            result = subprocess.run(
                ['Rscript', str(R_SCRIPT), str(R_INPUT), str(R_OUTPUT)],
                capture_output=True, text=True, timeout=300
            )

            print("  --- R Output ---")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"  {line}")
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if line.strip() and 'Warning' not in line:
                        print(f"  [R stderr] {line}")

            if result.returncode != 0:
                print(f"  ERROR: R script failed with return code {result.returncode}")
                print(f"  stderr: {result.stderr}")
                raise RuntimeError("R Cox regression failed")

            # ============================================================
            # Step 4.4: Load and display R results
            # ============================================================
            print("\n[Step 4.4] Loading R results...")
            results_df = pd.read_csv(R_OUTPUT)
            print(f"  Results: {len(results_df)} analyses")

            for _, row in results_df.iterrows():
                p_str = f"{row['P_value']:.4f}" if row['P_value'] >= 0.001 else "<0.001"
                print(f"  {row['Analysis']:20s} | {row['Database']:15s} | "
                      f"HR={row['HR']:.4f} ({row['CI_lower']:.4f}-{row['CI_upper']:.4f}) | "
                      f"P={p_str} | N={row['N']:,} | Events={row['Events']:,}")

            # Save formatted results table
            results_df.to_csv(TABLE_DIR / "TablePrimaryResults.csv", index=False)
            results_df.to_excel(TABLE_DIR / "TablePrimaryResults.xlsx", index=False)
            print(f"  Saved: Table/TablePrimaryResults.csv + .xlsx")

            # ============================================================
            # Step 4.5: Weighted Kaplan-Meier Survival Curves
            # ============================================================
            print("\n[Step 4.5] Generating weighted Kaplan-Meier curves...")
            from lifelines import KaplanMeierFitter

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Overall
            ax = axes[0, 0]
            for grp, label, color in [(0, 'Not Lonely', '#4C72B0'), (1, 'Lonely', '#DD8452')]:
                mask = df['loneliness'] == grp
                kmf = KaplanMeierFitter()
                kmf.fit(df.loc[mask, 'time_years'], df.loc[mask, 'event'],
                        weights=df.loc[mask, 'iptw'], label=label)
                kmf.plot_survival_function(ax=ax, color=color)
            ax.set_xlabel('Time (Years)')
            ax.set_ylabel('Survival Probability')
            ax.set_title('Overall (IPTW-Weighted)')
            ax.legend(fontsize=8)

            # Per-database
            db_list = [('HRS', axes[0, 1]), ('KLoSA', axes[1, 0]), ('SHARE', axes[1, 1])]
            for db_name, ax in db_list:
                db_df = df[df['database'] == db_name]
                wt_col = f'iptw_{db_name}' if f'iptw_{db_name}' in df.columns else 'iptw'
                for grp, label, color in [(0, 'Not Lonely', '#4C72B0'), (1, 'Lonely', '#DD8452')]:
                    mask = db_df['loneliness'] == grp
                    kmf = KaplanMeierFitter()
                    kmf.fit(db_df.loc[mask, 'time_years'], db_df.loc[mask, 'event'],
                            weights=db_df.loc[mask, wt_col], label=label)
                    kmf.plot_survival_function(ax=ax, color=color)
                ax.set_xlabel('Time (Years)')
                ax.set_ylabel('Survival Probability')
                ax.set_title(f'{db_name} (IPTW-Weighted)')
                ax.legend(fontsize=8)

            plt.tight_layout()
            fig.savefig(FIG_DIR / "FigureSurvivalCurves.png", dpi=300)
            fig.savefig(FIG_DIR / "FigureSurvivalCurves.pdf")
            plt.close()
            print(f"  Saved: FigureSurvivalCurves.png/pdf")

            # ============================================================
            # Step 4.6: Forest plot
            # ============================================================
            print("\n[Step 4.6] Generating forest plot...")
            plot_data = results_df[results_df['Analysis'].isin(['Per-Database', 'Meta-Analysis (RE)'])].copy()

            fig, ax = plt.subplots(figsize=(8, 4 + 0.6 * len(plot_data)))
            y_pos = np.arange(len(plot_data))

            # Plot points and CIs
            for i, (_, row) in enumerate(plot_data.iterrows()):
                color = '#C44E52' if row['Analysis'] == 'Meta-Analysis (RE)' else '#4C72B0'
                size = 120 if row['Analysis'] == 'Meta-Analysis (RE)' else 60
                shape = 's' if row['Analysis'] == 'Meta-Analysis (RE)' else 'o'
                ax.scatter(row['HR'], i, s=size, color=color, marker=shape, zorder=3)
                ax.hlines(i, row['CI_lower'], row['CI_upper'], colors=color, linewidth=2, zorder=2)

            # Null line
            ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, zorder=1)

            labels = []
            for _, row in plot_data.iterrows():
                if row['Analysis'] == 'Per-Database':
                    labels.append(f"{row['Database']} (N={row['N']:,})")
                else:
                    i2_val = row.get('I2', np.nan)
                    if pd.notna(i2_val):
                        labels.append(f"Overall RE (I2={i2_val:.1f}%)")
                    else:
                        labels.append("Overall RE")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Hazard Ratio (95% CI)')
            ax.set_title('Forest Plot: Loneliness → All-Cause Mortality')
            plt.tight_layout()
            fig.savefig(FIG_DIR / "FigureForestPlot.png", dpi=300)
            fig.savefig(FIG_DIR / "FigureForestPlot.pdf")
            plt.close()
            print(f"  Saved: FigureForestPlot.png/pdf")

            # ============================================================
            # Step 4.7: E-value calculation
            # ============================================================
            print("\n[Step 4.7] E-value calculation...")
            overall_hr = results_df.loc[results_df['Analysis'] == 'Overall', 'HR'].values[0]
            if overall_hr >= 1:
                e_value = overall_hr + np.sqrt(overall_hr * (overall_hr - 1))
            else:
                e_value = 1 / overall_hr + np.sqrt((1 / overall_hr) * ((1 / overall_hr) - 1))
            print(f"  Overall HR = {overall_hr:.4f}")
            print(f"  E-value = {e_value:.4f}")
            print(f"  Interpretation: Unmeasured confounder would need RR={e_value:.2f} with both treatment and outcome to explain away the effect")

            # Save E-value
            evalue_df = pd.DataFrame([{
                'Analysis': 'Overall',
                'HR': overall_hr,
                'E_value': round(e_value, 4)
            }])
            evalue_df.to_csv(TABLE_DIR / "TableEValue.csv", index=False)
            print(f"  Saved: Table/TableEValue.csv")

            # ============================================================
            # Step 4.8: Summary
            # ============================================================
            print("\n" + "=" * 70)
            print("Part 4 COMPLETE")
            print("=" * 70)

            overall = results_df[results_df['Analysis'] == 'Overall'].iloc[0]
            meta = results_df[results_df['Analysis'] == 'Meta-Analysis (RE)'].iloc[0]

            print(f"  Overall HR: {overall['HR']:.4f} ({overall['CI_lower']:.4f}-{overall['CI_upper']:.4f})")
            print(f"  Meta-analysis HR (RE): {meta['HR']:.4f} ({meta['CI_lower']:.4f}-{meta['CI_upper']:.4f})")
            i2 = meta.get('I2', np.nan)
            if pd.notna(i2):
                print(f"  Heterogeneity: I2={i2:.1f}%")
            print(f"  E-value: {e_value:.4f}")
            print(f"  Per-database analyses: HRS, KLoSA, SHARE")
            print(f"  Tables: TablePrimaryResults, TableEValue")
            print(f"  Figures: FigureSurvivalCurves, FigureForestPlot")

        finally:
            sys.stdout = original_stdout

    return results_df


if __name__ == "__main__":
    results = main()
