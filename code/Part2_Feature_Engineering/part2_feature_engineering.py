# ============================================================
# Part 2: Feature Engineering — T019 Loneliness → Mortality (IPTW)
# IPTW confounder matrix, disease_count, TableBaseline
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
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OTHER_DIR = OUTPUT_DIR / "intermediate"
TABLE_DIR = PROJECT_ROOT / "tables"
FIG_DIR = PROJECT_ROOT / "figures"

for d in [TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "Part2_result.txt"


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


def smd_continuous(x1, x0):
    m1, m0 = np.nanmean(x1), np.nanmean(x0)
    s1, s0 = np.nanvar(x1, ddof=1), np.nanvar(x0, ddof=1)
    pooled = np.sqrt((s1 + s0) / 2)
    return (m1 - m0) / pooled if pooled > 0 else 0.0


def smd_binary(x1, x0):
    p1, p0 = np.nanmean(x1), np.nanmean(x0)
    pooled = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return (p1 - p0) / pooled if pooled > 0 else 0.0


def main():
    with open(RESULT_FILE, 'w', encoding='utf-8') as rf:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.__stdout__, rf)
        try:
            print("=" * 70)
            print("Part 2: Feature Engineering — T019 IPTW")
            print("=" * 70)

            # ============================================================
            # Step 2.1: Load clean data
            # ============================================================
            print("\n[Step 2.1] Loading cleanDataPart1.csv...")
            df = pd.read_csv(OTHER_DIR / "cleanDataPart1.csv")
            print(f"  N={len(df):,}")

            # Step 2.1a: Load variable_name_mapping.csv
            mapping_df = pd.read_csv(DATA_DIR / "variable_name_mapping.csv")
            name_map = dict(zip(mapping_df['database_name'], mapping_df['sci_name']))
            print(f"  Loaded {len(name_map)} variable mappings")

            # ============================================================
            # Step 2.2: Create disease_count
            # ============================================================
            print("\n[Step 2.2] Creating disease_count...")
            disease_vars = ['hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre', 'arthre']
            df['disease_count'] = df[disease_vars].sum(axis=1)
            print(f"  disease_count distribution:")
            for v, c in df['disease_count'].value_counts().sort_index().items():
                print(f"    {v}: {c:,} ({c/len(df)*100:.1f}%)")

            # Update mapping with new variable (avoid duplicates)
            if 'disease_count' not in mapping_df['database_name'].values:
                new_mapping = pd.DataFrame([
                    ('disease_count', 'Chronic Disease Count', 'Number of chronic diseases (0-7)', 'count', '4_Disease_Burden', 8),
                ], columns=['database_name', 'sci_name', 'description', 'unit', 'sort_category', 'sort_priority'])
                mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)
                mapping_df.to_csv(DATA_DIR / "variable_name_mapping.csv", index=False)
            name_map['disease_count'] = 'Chronic Disease Count'
            print(f"  Mapping entries: {len(mapping_df)}")

            # ============================================================
            # Step 2.3: VIF Check
            # ============================================================
            print("\n[Step 2.3] VIF Check...")
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            confounders = ['age', 'gender', 'education', 'married',
                           'hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre', 'arthre',
                           'disease_count', 'depression', 'srh', 'adl', 'iadl',
                           'smoken', 'drink']

            X_vif = df[confounders].dropna()
            print(f"  VIF calculation on {len(X_vif):,} complete cases")

            vif_data = []
            for i, col in enumerate(confounders):
                vif = variance_inflation_factor(X_vif.values, i)
                vif_data.append((col, vif))
                if vif > 10:
                    vif_str = "inf" if np.isinf(vif) else f"{vif:.1f}"
                    print(f"  WARNING: {col} VIF={vif_str} (>10)")

            vif_df = pd.DataFrame(vif_data, columns=['variable', 'VIF'])
            print(f"\n  VIF Summary:")
            for _, row in vif_df.iterrows():
                flag = " (!)" if row['VIF'] > 10 else ""
                vif_str = "inf" if np.isinf(row['VIF']) else f"{row['VIF']:.2f}"
                print(f"    {row['variable']}: {vif_str}{flag}")

            # If disease_count has high VIF, remove it (perfect collinearity with individual diseases)
            dc_vif = vif_df[vif_df['variable'] == 'disease_count']['VIF'].values[0]
            if dc_vif > 10:
                vif_str = "inf" if np.isinf(dc_vif) else f"{dc_vif:.1f}"
                print(f"  Note: disease_count VIF={vif_str} (expected: sum of 7 disease vars)")
                print(f"  Action: Remove disease_count from confounders, keep individual diseases")
                confounders.remove('disease_count')

            # ============================================================
            # Step 2.4: Define IPTW confounders (final)
            # ============================================================
            print("\n[Step 2.4] Final IPTW confounders:")
            for i, c in enumerate(confounders):
                sci = name_map.get(c, c)
                print(f"  {i+1}. {c} -> {sci}")
            print(f"  Total: {len(confounders)} confounders")

            # Save confounder list
            pd.DataFrame({'confounder': confounders}).to_csv(DATA_DIR / "confounder_list.csv", index=False)

            # ============================================================
            # Step 2.5: Database indicator variables
            # ============================================================
            print("\n[Step 2.5] Database indicators...")
            for db in ['KLoSA', 'SHARE']:
                df[f'db_{db}'] = (df['database'] == db).astype(int)
                print(f"  db_{db}: {df[f'db_{db}'].sum():,} ({df[f'db_{db}'].mean()*100:.1f}%)")

            # ============================================================
            # Step 2.6: Generate TableBaseline (with SMD)
            # ============================================================
            print("\n[Step 2.6] Generating TableBaseline.xlsx...")
            import openpyxl
            from scipy import stats as sp_stats

            lonely = df[df['loneliness'] == 1]
            not_lonely = df[df['loneliness'] == 0]

            rows = []
            col_total = f'Total (N={len(df):,})'
            col_lonely = f'Lonely (N={len(lonely):,})'
            col_not = f'Not Lonely (N={len(not_lonely):,})'

            # --- Continuous variables: median [IQR] + Mann-Whitney U + SMD ---
            continuous_vars = ['age', 'depression', 'srh', 'adl', 'iadl', 'disease_count']
            print(f"\n  Continuous variables ({len(continuous_vars)}):")
            for var in continuous_vars:
                sci = name_map.get(var, var)
                total_med = df[var].median()
                total_q1, total_q3 = df[var].quantile([0.25, 0.75])
                l_med = lonely[var].median()
                l_q1, l_q3 = lonely[var].quantile([0.25, 0.75])
                nl_med = not_lonely[var].median()
                nl_q1, nl_q3 = not_lonely[var].quantile([0.25, 0.75])

                _, p = sp_stats.mannwhitneyu(lonely[var].dropna(), not_lonely[var].dropna(), alternative='two-sided')
                smd = smd_continuous(lonely[var].dropna().values, not_lonely[var].dropna().values)
                p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"

                rows.append({
                    'Characteristic': sci,
                    col_total: f"{total_med:.1f} [{total_q1:.1f}-{total_q3:.1f}]",
                    col_lonely: f"{l_med:.1f} [{l_q1:.1f}-{l_q3:.1f}]",
                    col_not: f"{nl_med:.1f} [{nl_q1:.1f}-{nl_q3:.1f}]",
                    'SMD': f"{smd:.3f}",
                    'P-value': p_str
                })
                print(f"    {sci}: SMD={smd:.3f}, P={p_str}")

            # --- Binary variables: n (%) + chi-square + SMD ---
            binary_vars = ['gender', 'married', 'hibpe', 'diabe', 'hearte', 'stroke',
                           'lunge', 'cancre', 'arthre', 'smoken', 'drink']
            print(f"\n  Binary variables ({len(binary_vars)}):")
            for var in binary_vars:
                sci = name_map.get(var, var)
                total_n = df[var].sum()
                total_pct = df[var].mean() * 100
                l_n = lonely[var].sum()
                l_pct = lonely[var].mean() * 100
                nl_n = not_lonely[var].sum()
                nl_pct = not_lonely[var].mean() * 100

                ct = pd.crosstab(df[var], df['loneliness'])
                chi2, p, dof, expected = sp_stats.chi2_contingency(ct)
                smd = smd_binary(lonely[var].dropna().values, not_lonely[var].dropna().values)
                p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"

                rows.append({
                    'Characteristic': sci,
                    col_total: f"{total_n:,.0f} ({total_pct:.1f}%)",
                    col_lonely: f"{l_n:,.0f} ({l_pct:.1f}%)",
                    col_not: f"{nl_n:,.0f} ({nl_pct:.1f}%)",
                    'SMD': f"{smd:.3f}",
                    'P-value': p_str
                })
                print(f"    {sci}: SMD={smd:.3f}, P={p_str}")

            # --- Categorical: education ---
            edu_map = {1: 'Low', 2: 'Medium', 3: 'High'}
            edu_first_idx = None
            print(f"\n  Education (categorical):")
            for edu_val, edu_label in edu_map.items():
                total_n = (df['education'] == edu_val).sum()
                l_n = (lonely['education'] == edu_val).sum()
                nl_n = (not_lonely['education'] == edu_val).sum()
                total_pct = total_n / len(df) * 100
                l_pct = l_n / len(lonely) * 100 if len(lonely) > 0 else 0
                nl_pct = nl_n / len(not_lonely) * 100 if len(not_lonely) > 0 else 0
                idx = len(rows)
                if edu_first_idx is None:
                    edu_first_idx = idx
                rows.append({
                    'Characteristic': f"Education: {edu_label}",
                    col_total: f"{total_n:,} ({total_pct:.1f}%)",
                    col_lonely: f"{l_n:,} ({l_pct:.1f}%)",
                    col_not: f"{nl_n:,} ({nl_pct:.1f}%)",
                    'SMD': '',
                    'P-value': ''
                })
                print(f"    {edu_label}: {total_n:,} ({total_pct:.1f}%)")

            # Education chi-square P-value + SMD (assign to first education row)
            ct_edu = pd.crosstab(df['education'], df['loneliness'])
            chi2_edu, p_edu, _, _ = sp_stats.chi2_contingency(ct_edu)
            p_edu_str = f"{p_edu:.3f}" if p_edu >= 0.001 else "<0.001"
            rows[edu_first_idx]['P-value'] = p_edu_str
            # Compute education SMD using ordinal encoding
            edu_smd = smd_continuous(lonely['education'].dropna().values, not_lonely['education'].dropna().values)
            rows[edu_first_idx]['SMD'] = f"{edu_smd:.3f}"
            print(f"    Education SMD={edu_smd:.3f}, P={p_edu_str}")

            # --- Outcome: mortality ---
            total_deaths = df['event'].sum()
            total_death_pct = df['event'].mean() * 100
            l_deaths = lonely['event'].sum()
            l_death_pct = lonely['event'].mean() * 100
            nl_deaths = not_lonely['event'].sum()
            nl_death_pct = not_lonely['event'].mean() * 100
            ct_death = pd.crosstab(df['event'], df['loneliness'])
            chi2_d, p_d, _, _ = sp_stats.chi2_contingency(ct_death)
            p_d_str = f"{p_d:.3f}" if p_d >= 0.001 else "<0.001"
            smd_death = smd_binary(lonely['event'].dropna().values, not_lonely['event'].dropna().values)

            rows.append({
                'Characteristic': 'All-Cause Mortality',
                col_total: f"{total_deaths:,} ({total_death_pct:.1f}%)",
                col_lonely: f"{l_deaths:,} ({l_death_pct:.1f}%)",
                col_not: f"{nl_deaths:,} ({nl_death_pct:.1f}%)",
                'SMD': f"{smd_death:.3f}",
                'P-value': p_d_str
            })
            print(f"\n  Mortality: SMD={smd_death:.3f}, P={p_d_str}")

            baseline_df = pd.DataFrame(rows)

            # Reorder columns: Characteristic, Total, Lonely, Not Lonely, SMD, P-value
            baseline_df = baseline_df[['Characteristic', col_total, col_lonely, col_not, 'SMD', 'P-value']]
            baseline_df.to_excel(TABLE_DIR / "TableBaseline.xlsx", index=False)
            print(f"\n  Saved: Table/TableBaseline.xlsx ({len(baseline_df)} rows)")

            # Print SMD summary
            smd_values = []
            for _, row in baseline_df.iterrows():
                try:
                    smd_values.append(float(row['SMD']))
                except (ValueError, TypeError):
                    pass
            if smd_values:
                print(f"  SMD range: {min(smd_values):.3f} to {max(smd_values):.3f}")
                print(f"  Variables with |SMD| > 0.1: {sum(abs(s) > 0.1 for s in smd_values)}/{len(smd_values)}")

            # ============================================================
            # Step 2.7: Save cleanDataPart2
            # ============================================================
            df.to_csv(OTHER_DIR / "cleanDataPart2.csv", index=False)
            print(f"\n[Step 2.7] Saved cleanDataPart2.csv ({len(df):,} rows, {len(df.columns)} cols)")

            # ============================================================
            # Step 2.8: Summary
            # ============================================================
            print("\n" + "=" * 70)
            print("Part 2 COMPLETE")
            print("=" * 70)
            print(f"  Confounders: {len(confounders)}")
            print(f"  disease_count created (0-7)")
            print(f"  VIF check: done (disease_count removed if VIF>10)")
            print(f"  TableBaseline.xlsx generated (with SMD column)")
            print(f"  Files: cleanDataPart2.csv, TableBaseline.xlsx, confounder_list.csv")

        finally:
            sys.stdout = original_stdout

    return df


if __name__ == "__main__":
    df = main()
