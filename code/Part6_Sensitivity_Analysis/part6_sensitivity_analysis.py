# ============================================================
# Part 6: Sensitivity Analyses — T019 Loneliness → Mortality (IPTW)
# Trimming, LOO, PS matching, E-value, per-DB balance
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

for d in [FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "Part6_result.txt"
R_SCRIPT = Path(__file__).parent / "part6_sensitivity.R"
R_INPUT = OTHER_DIR / "sensitivity_input.csv"
R_OUTPUT = OTHER_DIR / "sensitivity_results.csv"


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


def fmt_p(p):
    if pd.isna(p):
        return 'NA'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def compute_smd(x, treat, weights=None):
    x = np.asarray(x, dtype=float)
    t = np.asarray(treat).astype(bool)
    if weights is None:
        m1, m0 = x[t].mean(), x[~t].mean()
        v1, v0 = x[t].var(ddof=1), x[~t].var(ddof=1)
    else:
        w = np.asarray(weights, dtype=float)
        w1, w0 = w[t], w[~t]
        m1 = np.average(x[t], weights=w1)
        m0 = np.average(x[~t], weights=w0)
        v1 = np.average((x[t] - m1) ** 2, weights=w1)
        v0 = np.average((x[~t] - m0) ** 2, weights=w0)
    pooled = np.sqrt((v1 + v0) / 2.0)
    return (m1 - m0) / pooled if pooled > 0 else 0.0


def main():
    with open(RESULT_FILE, 'w', encoding='utf-8') as rf:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.__stdout__, rf)
        try:
            print("=" * 70)
            print("Part 6: Sensitivity Analyses — T019 IPTW")
            print("=" * 70)

            # ============================================================
            # Step 6.1: Load data
            # ============================================================
            print("\n[Step 6.1] Loading data...")
            df = pd.read_csv(OTHER_DIR / "cleanDataPart3.csv")
            mapping_df = pd.read_csv(DATA_DIR / "variable_name_mapping.csv")
            name_map = dict(zip(mapping_df['database_name'], mapping_df['sci_name']))
            print(f"  N={len(df):,}")

            # ============================================================
            # Step 6.2: E-value (with fixed CI)
            # ============================================================
            print("\n[Step 6.2] E-value analysis...")
            primary_results = pd.read_csv(OTHER_DIR / "cox_results.csv")
            overall_hr = primary_results.loc[primary_results['Analysis'] == 'Overall', 'HR'].values[0]
            overall_ci_lower = primary_results.loc[primary_results['Analysis'] == 'Overall', 'CI_lower'].values[0]
            overall_ci_upper = primary_results.loc[primary_results['Analysis'] == 'Overall', 'CI_upper'].values[0]

            if overall_hr >= 1:
                e_val_hr = overall_hr + np.sqrt(overall_hr * (overall_hr - 1))
                if overall_ci_lower >= 1:
                    e_val_ci = overall_ci_lower + np.sqrt(overall_ci_lower * (overall_ci_lower - 1))
                else:
                    e_val_ci = 1.0
            else:
                rr = 1.0 / overall_hr
                e_val_hr = rr + np.sqrt(rr * (rr - 1))
                rr_ci = 1.0 / overall_ci_upper if overall_ci_upper > 0 else np.inf
                if rr_ci >= 1:
                    e_val_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
                else:
                    e_val_ci = 1.0

            print(f"  Primary HR = {overall_hr:.4f}")
            print(f"  E-value (point) = {e_val_hr:.4f}")
            print(f"  E-value (CI) = {e_val_ci:.4f}")

            # ============================================================
            # Step 6.3: Per-database SMD balance
            # ============================================================
            print("\n[Step 6.3] Per-database SMD balance...")
            confounder_list = pd.read_csv(DATA_DIR / "confounder_list.csv")['confounder'].tolist()

            per_db_smd_rows = []
            for db in ['HRS', 'KLoSA', 'SHARE']:
                db_df = df[df['database'] == db]
                wt_col = f'iptw_{db}'
                if wt_col not in db_df.columns:
                    continue
                for feat in confounder_list:
                    smd_pre = compute_smd(db_df[feat].values, db_df['loneliness'].values)
                    smd_post = compute_smd(db_df[feat].values, db_df['loneliness'].values,
                                           weights=db_df[wt_col].values)
                    per_db_smd_rows.append({
                        'Database': db, 'Variable': feat,
                        'sci_name': name_map.get(feat, feat),
                        'SMD_unweighted': round(smd_pre, 4),
                        'SMD_weighted': round(smd_post, 4),
                        'balanced': 'Yes' if abs(smd_post) < 0.1 else 'No'
                    })

            per_db_smd_df = pd.DataFrame(per_db_smd_rows)
            per_db_smd_df.to_csv(TABLE_DIR / "TablePerDBSMDBalance.csv", index=False)

            for db in ['HRS', 'KLoSA', 'SHARE']:
                db_smd = per_db_smd_df[per_db_smd_df['Database'] == db]
                n_unbal = (db_smd['balanced'] == 'No').sum()
                print(f"  {db}: {n_unbal}/{len(db_smd)} |SMD| >= 0.1")

            print(f"  Saved: Table/TablePerDBSMDBalance.csv")

            # ============================================================
            # Step 6.4: Run R sensitivity analyses
            # ============================================================
            print("\n[Step 6.4] Running R sensitivity analyses...")

            r_cols = ['database', 'loneliness', 'event', 'time_years', 'iptw', 'ps_score']
            r_cols += [c for c in confounder_list if c in df.columns]
            df[r_cols].to_csv(R_INPUT, index=False)

            result = subprocess.run(
                ['Rscript', str(R_SCRIPT), str(R_INPUT), str(R_OUTPUT)],
                capture_output=True, text=True, timeout=600
            )

            print("  --- R Output ---")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"  {line}")
            if result.stderr:
                stderr_lines = [l for l in result.stderr.split('\n')
                                if l.strip() and 'Loading' not in l and 'Type' not in l
                                and 'deprecated' not in l and 'glmnet' not in l]
                for line in stderr_lines:
                    print(f"  [R] {line}")

            if result.returncode != 0:
                print(f"  WARNING: R returned code {result.returncode}")

            # ============================================================
            # Step 6.5: Display and save results
            # ============================================================
            print("\n[Step 6.5] Sensitivity results summary...")
            if os.path.exists(R_OUTPUT):
                sens_df = pd.read_csv(R_OUTPUT)

                # Format P-values
                sens_df['P_formatted'] = sens_df['P'].apply(fmt_p)

                sens_df.to_csv(TABLE_DIR / "TableSensitivityResults.csv", index=False)
                sens_df.to_excel(TABLE_DIR / "TableSensitivityResults.xlsx", index=False)

                for _, row in sens_df.iterrows():
                    print(f"  {row['Analysis']:30s} | HR={row['HR']:.4f} "
                          f"({row['CI_lower']:.4f}-{row['CI_upper']:.4f}) | "
                          f"P={row['P_formatted']}")

                # Forest plot
                print("\n[Step 6.6] Generating sensitivity forest plot...")
                fig, ax = plt.subplots(figsize=(10, 3 + 0.5 * len(sens_df)))
                y_pos = np.arange(len(sens_df))

                for i, (_, row) in enumerate(sens_df.iterrows()):
                    color = '#C44E52' if i == 0 else '#4C72B0'
                    size = 100 if i == 0 else 50
                    ax.scatter(row['HR'], i, s=size, color=color, zorder=3)
                    ax.hlines(i, row['CI_lower'], row['CI_upper'], colors=color, linewidth=1.5, zorder=2)

                ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, zorder=1)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(sens_df['Analysis'].values, fontsize=8)
                ax.set_xlabel('Hazard Ratio (95% CI)')
                ax.set_title('Sensitivity Analyses: Loneliness and All-Cause Mortality')
                plt.tight_layout()
                fig.savefig(FIG_DIR / "FigureSensitivityForest.png", dpi=300)
                fig.savefig(FIG_DIR / "FigureSensitivityForest.pdf")
                plt.close()
                print(f"  Saved: FigureSensitivityForest.png/pdf")
                print(f"  Saved: Table/TableSensitivityResults.csv + .xlsx")
            else:
                print("  WARNING: R results not found")

            # ============================================================
            # Step 6.7: MICE note
            # ============================================================
            print("\n[Step 6.7] MICE comparison note:")
            print("  Data was already MICE-imputed in Part1 (IterativeImputer per-database).")
            print("  No missing values remain in confounders — complete-case vs MICE not applicable.")

            # ============================================================
            # Step 6.8: Summary
            # ============================================================
            print("\n" + "=" * 70)
            print("Part 6 COMPLETE")
            print("=" * 70)
            print(f"  E-value: {e_val_hr:.4f} (point), {e_val_ci:.4f} (CI)")
            print(f"  Per-database SMD: TablePerDBSMDBalance.csv")
            print(f"  Sensitivity analyses: trimming, LOO, PS matching, alternate PS")
            print(f"  Tables: TableSensitivityResults, TablePerDBSMDBalance")
            print(f"  Figures: FigureSensitivityForest")

        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
