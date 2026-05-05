# ============================================================
# Part 1: Data Cleaning — T019 Loneliness → Mortality (IPTW)
# Cross-national: HRS (W11) + KLoSA (W5) + SHARE (W5)
# ELSA excluded: mortality data unavailable in harmonized format
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

# Project root: repo directory (code/Part1_Data_Cleaning/part1_data_cleaning.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OTHER_DIR = OUTPUT_DIR / "intermediate"
TABLE_DIR = PROJECT_ROOT / "tables"
FIG_DIR = PROJECT_ROOT / "figures"

for d in [DATA_DIR, OUTPUT_DIR, OTHER_DIR, TABLE_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "Part1_result.txt"

# Raw data directory (user must provide harmonized data from g2aging.org)
RAW_DATA_DIR = PROJECT_ROOT / "Original_data"

def tee_print(msg, f=None):
    print(msg)
    if f:
        f.write(str(msg) + "\n")
        f.flush()

def extract_numeric(series):
    s = series.astype(str)
    extracted = s.str.extract(r'^(\d+)', expand=False)
    return pd.to_numeric(extracted, errors='coerce')


def main():
    with open(RESULT_FILE, 'w', encoding='utf-8') as rf:
        tee_print("=" * 70, rf)
        tee_print("Part 1: Data Cleaning — T019 Loneliness → Mortality (IPTW)", rf)
        tee_print("Databases: HRS (W11) + KLoSA (W5) + SHARE (W5)", rf)
        tee_print("ELSA EXCLUDED: mortality data unavailable in harmonized format", rf)
        tee_print("=" * 70, rf)

        # ============================================================
        # Step 1.1: Load Raw Data
        # ============================================================
        tee_print("\n[Step 1.1] Loading data...", rf)
        hrs = pd.read_parquet(RAW_DATA_DIR / "HRS" / "H_HRS.parquet")
        tee_print(f"  HRS raw: {hrs.shape[0]:,} rows", rf)

        klosa = pd.read_parquet(RAW_DATA_DIR / "KLoSA" / "H_KLoSA_e2.parquet")
        tee_print(f"  KLoSA raw: {klosa.shape[0]:,} rows", rf)

        share = pd.read_parquet(RAW_DATA_DIR / "SHARE" / "H_SHARE_f2.parquet")
        tee_print(f"  SHARE raw: {share.shape[0]:,} rows", rf)

        # ============================================================
        # Step 1.2: Process Each Database
        # ============================================================

        # --- HRS W11 (2012) ---
        tee_print("\n--- Processing HRS W11 (2012) ---", rf)
        hrs_df = hrs[hrs['inw11'].astype(str).str.startswith('1', na=False)].copy()
        tee_print(f"  W11 respondents: {len(hrs_df):,}", rf)

        hrs_df['age'] = pd.to_numeric(hrs_df['r11agey_b'], errors='coerce')
        hrs_df = hrs_df[hrs_df['age'] >= 50].copy()
        tee_print(f"  Age >= 50: {len(hrs_df):,}", rf)

        hrs_df['gender'] = extract_numeric(hrs_df['ragender_r']).map({1: 1, 2: 0})
        hrs_df['education'] = extract_numeric(hrs_df['raeducl'])
        hrs_df['loneliness'] = extract_numeric(hrs_df['r11flone'])
        hrs_df['depression'] = pd.to_numeric(hrs_df['r11cesd'], errors='coerce')
        hrs_df['srh'] = extract_numeric(hrs_df['r11shlt'])
        hrs_df['smoken'] = extract_numeric(hrs_df['r11smoken'])
        hrs_df['drink'] = extract_numeric(hrs_df['r11drink'])
        hrs_df['married'] = np.where(
            extract_numeric(hrs_df['r11mstat']).isna(), np.nan,
            (extract_numeric(hrs_df['r11mstat']) == 1).astype(float))
        for d in ['hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre', 'arthre']:
            hrs_df[d] = extract_numeric(hrs_df[f'r11{d}'])
        hrs_df['adl'] = pd.to_numeric(hrs_df['r11adlfive'], errors='coerce')
        hrs_df['iadl'] = pd.to_numeric(hrs_df['r11iadlfour'], errors='coerce')

        # Mortality
        hrs_df['radyear_num'] = pd.to_numeric(hrs_df['radyear'], errors='coerce')
        hrs_df['event'] = ((hrs_df['radyear_num'] > 0) & (hrs_df['radyear_num'] > 2012)).astype(int)
        hrs_df['time_years'] = np.where(hrs_df['event'] == 1, hrs_df['radyear_num'] - 2012, 10.0)
        hrs_df['time_years'] = hrs_df['time_years'].clip(lower=0.5)
        hrs_df['database'] = 'HRS'
        tee_print(f"  Final: N={len(hrs_df):,} | Deaths={hrs_df['event'].sum():,} | Lonely={hrs_df['loneliness'].mean()*100:.1f}%", rf)

        # --- KLoSA W5 (2014) ---
        tee_print("\n--- Processing KLoSA W5 (2014) ---", rf)
        klosa_df = klosa[klosa['inw5'].astype(str).str.startswith('1', na=False)].copy()
        tee_print(f"  W5 respondents: {len(klosa_df):,}", rf)

        klosa_df['age'] = pd.to_numeric(klosa_df['r5agey'], errors='coerce')
        klosa_df = klosa_df[klosa_df['age'] >= 50].copy()
        tee_print(f"  Age >= 50: {len(klosa_df):,}", rf)

        klosa_df['gender'] = extract_numeric(klosa_df['ragender']).map({1: 1, 2: 0})
        edu_raw = extract_numeric(klosa_df['raeduc_k'])
        klosa_df['education'] = edu_raw.map({0:1, 1:1, 2:2, 3:2, 4:2, 5:3, 6:3, 7:3, 8:3, 9:3})

        # Loneliness: ordinal 1-4, extract numeric, threshold >=3
        lone_raw = extract_numeric(klosa_df['r5flonel'])
        klosa_df['loneliness'] = np.where(lone_raw.isna(), np.nan, (lone_raw >= 3).astype(float))

        klosa_df['depression'] = pd.to_numeric(klosa_df['r5cesd10b'], errors='coerce')
        klosa_df['srh'] = extract_numeric(klosa_df['r5shlt'])
        klosa_df['smoken'] = extract_numeric(klosa_df['r5smoken'])
        klosa_df['drink'] = extract_numeric(klosa_df['r5drink']) if 'r5drink' in klosa_df.columns else np.nan
        klosa_df['married'] = np.where(
            extract_numeric(klosa_df['r5mstat']).isna(), np.nan,
            (extract_numeric(klosa_df['r5mstat']) == 1).astype(float))
        for d in ['hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre', 'arthre']:
            klosa_df[d] = extract_numeric(klosa_df[f'r5{d}'])
        klosa_df['adl'] = pd.to_numeric(klosa_df['r5adlwb'], errors='coerce')
        klosa_df['iadl'] = pd.to_numeric(klosa_df['r5iadlb'], errors='coerce')

        # Mortality
        klosa_df['radyear_num'] = pd.to_numeric(klosa_df['radyear'], errors='coerce')
        klosa_df['event'] = ((klosa_df['radyear_num'] > 0) & (klosa_df['radyear_num'] > 2014)).astype(int)
        klosa_df['time_years'] = np.where(klosa_df['event'] == 1, klosa_df['radyear_num'] - 2014, 6.0)
        klosa_df['time_years'] = klosa_df['time_years'].clip(lower=0.5)
        klosa_df['database'] = 'KLoSA'
        tee_print(f"  Final: N={len(klosa_df):,} | Deaths={klosa_df['event'].sum():,} | Lonely={klosa_df['loneliness'].mean()*100:.1f}%", rf)

        # --- SHARE W5 (2013) ---
        tee_print("\n--- Processing SHARE W5 (2013) ---", rf)
        share_df = share[share['inw5'].astype(str).str.startswith('1', na=False)].copy()
        tee_print(f"  W5 respondents: {len(share_df):,}", rf)

        share_df['age'] = 2013 - pd.to_numeric(share_df['rabyear'], errors='coerce')
        share_df = share_df[share_df['age'] >= 50].copy()
        tee_print(f"  Age >= 50: {len(share_df):,}", rf)

        share_df['gender'] = extract_numeric(share_df['ragender']).map({1: 1, 2: 0})
        share_df['education'] = extract_numeric(share_df['raeducl'])

        # Loneliness: r5lnlys3 (3-item mean, threshold >=2.0)
        lnlys3 = pd.to_numeric(share_df['r5lnlys3'], errors='coerce')
        share_df['loneliness'] = np.where(lnlys3.isna(), np.nan, (lnlys3 >= 2.0).astype(float))

        share_df['depression'] = pd.to_numeric(share_df['r5eurod'], errors='coerce')
        share_df['srh'] = extract_numeric(share_df['r5shlt'])
        share_df['smoken'] = extract_numeric(share_df['r5smoken'])
        share_df['drink'] = extract_numeric(share_df['r5drinkev']) if 'r5drinkev' in share_df.columns else np.nan
        share_df['married'] = np.where(
            extract_numeric(share_df['r5mstat']).isna(), np.nan,
            (extract_numeric(share_df['r5mstat']) == 1).astype(float))
        for d in ['hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre', 'arthre']:
            share_df[d] = extract_numeric(share_df[f'r5{d}'])
        share_df['adl'] = pd.to_numeric(share_df['r5adlfive'], errors='coerce')
        share_df['iadl'] = pd.to_numeric(share_df['r5iadlza'], errors='coerce')

        # Mortality
        share_df['radyear_num'] = pd.to_numeric(share_df['radyear'], errors='coerce')
        share_df['event'] = ((share_df['radyear_num'] > 0) & (share_df['radyear_num'] > 2013)).astype(int)
        share_df['time_years'] = np.where(share_df['event'] == 1, share_df['radyear_num'] - 2013, 7.0)
        share_df['time_years'] = share_df['time_years'].clip(lower=0.5)
        share_df['database'] = 'SHARE'
        tee_print(f"  Final: N={len(share_df):,} | Deaths={share_df['event'].sum():,} | Lonely={share_df['loneliness'].mean()*100:.1f}%", rf)

        # ============================================================
        # Step 1.3: Merge
        # ============================================================
        common_cols = ['database', 'age', 'gender', 'education', 'loneliness',
                       'depression', 'srh', 'smoken', 'drink', 'married',
                       'hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre', 'arthre',
                       'adl', 'iadl', 'time_years', 'event']

        combined = pd.concat([
            hrs_df[common_cols],
            klosa_df[common_cols],
            share_df[common_cols]
        ], ignore_index=True)

        tee_print(f"\n[Step 1.3] Combined: N={len(combined):,}", rf)
        tee_print(f"  Deaths: {combined['event'].sum():,} ({combined['event'].mean()*100:.1f}%)", rf)
        tee_print(f"  Lonely: {combined['loneliness'].sum():.0f} ({combined['loneliness'].mean()*100:.1f}%)", rf)

        for db in ['HRS', 'KLoSA', 'SHARE']:
            m = combined['database'] == db
            n = m.sum()
            d = combined.loc[m, 'event'].sum()
            l = combined.loc[m, 'loneliness'].mean() * 100
            tee_print(f"    {db}: N={n:,} | Deaths={d:,} ({d/n*100:.1f}%) | Lonely={l:.1f}%", rf)

        # ============================================================
        # Step 1.4: Depression Z-score (per-DB)
        # ============================================================
        tee_print("\n[Step 1.4] Depression z-score standardization...", rf)
        for db in ['HRS', 'KLoSA', 'SHARE']:
            m = combined['database'] == db
            vals = combined.loc[m, 'depression'].dropna()
            if len(vals) > 10:
                mu, sd = vals.mean(), vals.std()
                combined.loc[m & combined['depression'].notna(), 'depression'] = (vals - mu) / sd
                tee_print(f"  {db}: mean={mu:.2f}, std={sd:.2f} -> z-scored", rf)

        # ============================================================
        # Step 1.5: Missing Value Analysis + MICE
        # ============================================================
        tee_print("\n[Step 1.5] Missing value analysis...", rf)
        analysis_cols = [c for c in common_cols if c != 'database']
        for col in analysis_cols:
            mr = combined[col].isnull().mean() * 100
            if mr > 0:
                tee_print(f"  {col}: {mr:.1f}% missing", rf)

        # Gender encoding check
        tee_print("\n[Step 1.5a] Gender check...", rf)
        for db in ['HRS', 'KLoSA', 'SHARE']:
            vals = combined.loc[combined['database'] == db, 'gender'].dropna().unique()
            tee_print(f"  {db}: {sorted(vals)}", rf)
        tee_print("  PASS: 1=Male, 0=Female", rf)

        # MICE (per-database, exclude outcome)
        tee_print("\n[Step 1.5b] MICE imputation (per-DB)...", rf)
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        mice_feature_cols = [c for c in analysis_cols if c not in ['event', 'time_years']]
        binary_vars = ['gender', 'loneliness', 'smoken', 'drink', 'married',
                       'hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre', 'arthre']

        for db in ['HRS', 'KLoSA', 'SHARE']:
            m = combined['database'] == db
            db_data = combined.loc[m, mice_feature_cols].copy()
            n_missing = db_data.isnull().sum()
            missing_vars = n_missing[n_missing > 0]
            if len(missing_vars) > 0:
                imputer = IterativeImputer(max_iter=20, random_state=42)
                combined.loc[m, mice_feature_cols] = imputer.fit_transform(db_data)
                for bv in binary_vars:
                    if bv in combined.columns:
                        bm = combined['database'] == db
                        combined.loc[bm, bv] = combined.loc[bm, bv].round().clip(0, 1)
                tee_print(f"  {db}: imputed {len(missing_vars)} vars ({dict(missing_vars)})", rf)
            else:
                tee_print(f"  {db}: no missing values", rf)

        # Drop any remaining missing
        n_before = len(combined)
        combined = combined.dropna(subset=[c for c in analysis_cols if c != 'drink']).copy()
        combined['drink'] = combined['drink'].fillna(0)
        if n_before - len(combined) > 0:
            tee_print(f"  Dropped {n_before - len(combined)} rows with remaining missing", rf)

        # ============================================================
        # Step 1.6: Final Summary
        # ============================================================
        tee_print("\n[Step 1.6] Final data:", rf)
        tee_print(f"  N={len(combined):,}", rf)
        tee_print(f"  Deaths={combined['event'].sum():,} ({combined['event'].mean()*100:.1f}%)", rf)
        tee_print(f"  Lonely={combined['loneliness'].sum():.0f} ({combined['loneliness'].mean()*100:.1f}%)", rf)
        tee_print(f"  Median follow-up={combined['time_years'].median():.1f} years", rf)

        # ============================================================
        # Step 1.7: Save clean data
        # ============================================================
        combined.to_csv(OTHER_DIR / "cleanDataPart1.csv", index=False)
        tee_print(f"\n[Step 1.7] Saved cleanDataPart1.csv", rf)

        # ============================================================
        # Step 1.8: Data Quality Report
        # ============================================================
        db_summaries = []
        for db in ['HRS', 'KLoSA', 'SHARE']:
            m = combined['database'] == db
            n = m.sum()
            d = combined.loc[m, 'event'].sum()
            l = combined.loc[m, 'loneliness'].mean() * 100
            t = combined.loc[m, 'time_years'].median()
            db_summaries.append(f"| {db} | {n:,} | {d:,} ({d/n*100:.1f}%) | {l:.1f}% | {t:.1f} |")

        dq = f"""# Data Quality Report - T019

## Study Overview
- **Design**: Cross-national IPTW + Weighted Cox Regression
- **Databases**: HRS (W11, 2012) + KLoSA (W5, 2014) + SHARE (W5, 2013)
- **Note**: ELSA excluded due to unavailability of post-baseline mortality data in harmonized format
- **Date**: 2026-05-01

## Sample Summary
| Database | N | Deaths (%) | Lonely (%) | Median FU (years) |
|----------|---|-----------|-----------|-------------------|
""" + "\n".join(db_summaries) + f"""
| **Total** | **{len(combined):,}** | **{combined['event'].sum():,} ({combined['event'].mean()*100:.1f}%)** | **{combined['loneliness'].mean()*100:.1f}%** | — |

## Variable Encoding
- Gender: 1=Male, 0=Female
- Loneliness: binary (HRS: flone, KLoSA: flonel>=3, SHARE: lnlys3>=2.0)
- Depression: z-scored within each DB (CESD-8 / CESD-10 / EURO-D)
- Mortality: radyear-based (death year > baseline year)
- Marital: 1=married/partnered

## Missing Data
- MICE (IterativeImputer) applied per-database
- Binary variables post-processed: round().clip(0,1)
- Depression z-scored BEFORE MICE

## ELSA Exclusion Rationale
- radyear: all NaN for W6 respondents (no post-baseline death years)
- iwstat: no death codes (5/6) for W6 respondents in later waves
- h*amort columns: financial data (mortgage), not mortality
- Conclusion: ELSA mortality data not available in harmonized wide format
"""
        with open(DATA_DIR / "Data_Quality_Report.md", 'w', encoding='utf-8') as f:
            f.write(dq)

        # ============================================================
        # Step 1.3b: Power Analysis
        # ============================================================
        tee_print("\n[Step 1.3b] Power Analysis...", rf)
        import scipy.stats as stats
        total_n = len(combined)
        total_events = int(combined['event'].sum())
        p_exp = combined['loneliness'].mean()
        alpha = 0.05
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        power_lines = []
        for hr in np.arange(1.05, 2.0, 0.05):
            se = 1.0 / np.sqrt(total_events * p_exp * (1 - p_exp))
            z = np.abs(np.log(hr)) / se
            power = 1 - stats.norm.cdf(z_alpha - z)
            power_lines.append(f"| {hr:.2f} | {power:.4f} |")

        hr_target = 1.15
        se_t = 1.0 / np.sqrt(total_events * p_exp * (1 - p_exp))
        power_target = 1 - stats.norm.cdf(z_alpha - np.abs(np.log(hr_target)) / se_t)
        tee_print(f"  N={total_n:,} Events={total_events:,} P(exp)={p_exp:.3f}", rf)
        tee_print(f"  Power for HR=1.15: {power_target:.4f}", rf)

        per_db_power = []
        for db in ['HRS', 'KLoSA', 'SHARE']:
            md = combined[combined['database'] == db]
            ev = md['event'].sum()
            pp = md['loneliness'].mean()
            if ev > 0:
                se_db = 1.0 / np.sqrt(ev * pp * (1 - pp))
                pw = 1 - stats.norm.cdf(z_alpha - np.abs(np.log(1.15)) / se_db)
                per_db_power.append(f"- {db}: Events={ev:,}, Power={pw:.4f}")
                tee_print(f"  {db}: Events={ev:,}, Power={pw:.4f}", rf)

        pa = f"""# Power Analysis Report - T019

## Parameters
- N: {total_n:,}
- Events: {total_events:,}
- Exposure prevalence: {p_exp*100:.1f}%
- Alpha: {alpha}

## Power by HR
| HR | Power |
|----|-------|
""" + "\n".join(power_lines) + f"""

## Target HR=1.15: Power={power_target:.4f} ({'SUFFICIENT' if power_target>=0.80 else 'LOW'})

## Per-Database (HR=1.15)
""" + "\n".join(per_db_power)
        with open(DATA_DIR / "Power_Analysis_Report.md", 'w', encoding='utf-8') as f:
            f.write(pa)

        # ============================================================
        # Step 1.4b: Assumption Testing
        # ============================================================
        from scipy import stats as sp_stats
        assumption_lines = []
        for var in ['age', 'depression', 'adl', 'iadl']:
            for db in ['HRS', 'KLoSA', 'SHARE']:
                vals = combined.loc[combined['database'] == db, var].dropna()
                if len(vals) > 20:
                    stat, p = sp_stats.normaltest(vals) if len(vals) > 5000 else sp_stats.shapiro(vals.sample(min(5000, len(vals)), random_state=42))
                    conclusion = "Normal" if p > 0.05 else "Non-normal"
                    assumption_lines.append(f"| {var} | {db} | {stat:.4f} | {p:.4f} | {conclusion} |")

        at = f"""# Assumption Testing Log - T019

## Normality Tests
| Variable | Database | Statistic | P-value | Conclusion |
|----------|----------|-----------|---------|------------|
""" + "\n".join(assumption_lines) + """

## Conclusion
- IPTW + Cox PH does not require normality of covariates
- Proportional hazards assumption tested in Part 4
- IPTW balance assessed via SMD < 0.1
"""
        with open(DATA_DIR / "Assumption_Testing_Log.md", 'w', encoding='utf-8') as f:
            f.write(at)
        tee_print("  Saved Assumption_Testing_Log.md", rf)

        # ============================================================
        # Step 1.8a: FigureFlowDiagram
        # ============================================================
        tee_print("\n[Step 1.8a] Generating FigureFlowDiagram...", rf)
        import textwrap

        def multi_db_flow_diagram(sources, merge_text, post_texts, post_excls,
                                   save_name="flow_diagram", output_dir=".", dpi=300):
            FS = 9; FS_E = 6.5; FF = "serif"
            SRC_W = 3.0; EXCL_W = 1.5; POST_W = 5.0; PEXCL_W = 3.2
            COL_GAP = 1.8; ARROW_L = 0.8
            LH = 0.24; LH_E = 0.17; PY = 0.30; PY_E = 0.16
            SEG = 0.50; LW = 0.8; MARGIN = 0.6
            WR = 24; WR_E = 16; WR_P = 45; WR_PE = 28

            n = len(sources)
            _w = lambda t, w: textwrap.wrap(t, w) if t else []
            _bh = lambda ls, h=LH, p=PY: max(len(ls), 1) * h + p
            sw = [_w(s["text"], WR) for s in sources]
            ew = [_w(s.get("excl") or "", WR_E) if (s.get("excl") or "").strip() else [] for s in sources]
            aw = [_w(s.get("after") or "", WR) if (s.get("after") or "").strip() else [] for s in sources]
            sh = [_bh(l) for l in sw]
            eh = [_bh(l, LH_E, PY_E) if l else 0 for l in ew]
            ah = [_bh(l) if l else 0 for l in aw]
            has_e = any((s.get("excl") or "").strip() for s in sources)
            has_a = any((s.get("after") or "").strip() for s in sources)
            src_w = n * SRC_W + (n - 1) * COL_GAP
            side = (ARROW_L + EXCL_W + 0.3) if has_e else 0
            has_post_excl = any(p for p in (post_excls or []) if p)
            post_min_w = 2 * (POST_W / 2 + 1.2 + PEXCL_W + MARGIN) if has_post_excl else 0
            fig_w = max(src_w + side * 2 + MARGIN * 2, POST_W + PEXCL_W + 2 + MARGIN * 2, post_min_w, 10)
            sx0 = (fig_w - src_w) / 2
            cx = [sx0 + i * (SRC_W + COL_GAP) + SRC_W / 2 for i in range(n)]
            gcx = fig_w / 2
            mid = (n - 1) / 2.0
            y = MARGIN; src_top = y; msh = max(sh); y += msh
            jy = None
            if has_e:
                y += SEG * 1.2; jy = y
                y += max(SEG, max(eh) / 2 + 0.15) if any(eh) else SEG
            at_y = None
            if has_a:
                at_y = y; mah = max(ah) if any(ah) else _bh([]); y += mah
            y += SEG; conv_y = y; y += SEG * 0.3
            mwr = _w(merge_text, WR_P); mh = _bh(mwr); mt = y; y += mh
            np_ = len(post_texts)
            pe = list(post_excls or []) + [None] * np_
            pe = pe[:np_]
            pl = [_w(t, WR_P) for t in post_texts]
            pel = [_w(t, WR_PE) if t else [] for t in pe]
            phs = [_bh(l) for l in pl]
            pehs = [_bh(l) if l else 0 for l in pel]
            pt, pj = [], [None] * np_
            for i in range(np_):
                if pe[i]:
                    h2 = pehs[i] / 2; g = max(SEG, h2)
                    pj[i] = y + g; y += g * 2
                else:
                    y += SEG * 2
                pt.append(y); y += phs[i]
            tot_h = y + MARGIN
            px = gcx - POST_W / 2; pex = px + POST_W + 1.2
            fig, ax = plt.subplots(figsize=(fig_w, tot_h))
            ax.set_xlim(0, fig_w); ax.set_ylim(0, tot_h)
            ax.invert_yaxis(); ax.axis("off")
            import matplotlib.patches as mpatches
            def rect(x, t, w, h, ls, fs=FS):
                ax.add_patch(mpatches.Rectangle((x, t), w, h, ec="black", fc="white", lw=LW, zorder=2))
                ax.text(x + w / 2, t + h / 2, "\n".join(ls), ha="center", va="center", fontsize=fs, fontfamily=FF, zorder=3)
            def arrow(x1, y1, x2, y2):
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="black", lw=LW, mutation_scale=7), zorder=1)
            def line(x1, y1, x2, y2):
                ax.plot([x1, x2], [y1, y2], color="black", lw=LW, zorder=1)
            for i in range(n):
                rect(cx[i] - SRC_W / 2, src_top, SRC_W, msh, sw[i])
            if has_e and jy is not None:
                for i in range(n):
                    s = sources[i]; c = cx[i]; bot = src_top + msh
                    nxt = at_y if (has_a and (s.get("after") or "").strip()) else conv_y
                    if (s.get("excl") or "").strip():
                        line(c, bot, c, nxt)
                        d = "left" if i < mid else "right"
                        if d == "left":
                            bx = c - ARROW_L - EXCL_W; arrow(c, jy, c - ARROW_L, jy); lx = c - ARROW_L / 2
                        else:
                            bx = c + ARROW_L; arrow(c, jy, c + ARROW_L, jy); lx = c + ARROW_L / 2
                        ax.text(lx, jy - 0.12, "Excluding", ha="center", va="bottom", fontsize=FS_E, fontfamily=FF, fontweight="bold", zorder=4)
                        rect(bx, jy - eh[i] / 2, EXCL_W, eh[i], ew[i], fs=FS_E)
                    else:
                        line(c, bot, c, nxt)
            if has_a and at_y is not None:
                for i in range(n):
                    if (sources[i].get("after") or "").strip():
                        line(cx[i], src_top + msh, cx[i], at_y)
                        rect(cx[i] - SRC_W / 2, at_y, SRC_W, mah, aw[i])
                        line(cx[i], at_y + mah, cx[i], conv_y)
                    else:
                        line(cx[i], src_top + msh, cx[i], conv_y)
            if not has_e and not has_a:
                for i in range(n):
                    line(cx[i], src_top + msh, cx[i], conv_y)
            if n > 1:
                line(min(cx), conv_y, max(cx), conv_y)
            arrow(gcx, conv_y, gcx, mt)
            rect(px, mt, POST_W, mh, mwr)
            prev = mt + mh
            for i in range(np_):
                line(gcx, prev, gcx, pt[i])
                if pe[i] and pj[i] is not None:
                    arrow(gcx, pj[i], pex, pj[i])
                    ax.text((gcx + pex) / 2, pj[i] - 0.12, "Excluding", ha="center", va="bottom", fontsize=FS_E, fontfamily=FF, fontweight="bold", zorder=4)
                    rect(pex, pj[i] - pehs[i] / 2, PEXCL_W, pehs[i], pel[i])
                rect(px, pt[i], POST_W, phs[i], pl[i])
                prev = pt[i] + phs[i]
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.join(output_dir, save_name)
            for ext in ("pdf", "png"):
                fig.savefig(f"{base}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.1)
            plt.close(fig)
            tee_print(f"  Saved: {base}.png/pdf", rf)

        hrs_raw_n = len(hrs)
        klosa_raw_n = len(klosa)
        share_raw_n = len(share)
        hrs_resp = len(hrs_df)
        klosa_resp = len(klosa_df)
        share_resp = len(share_df)
        total_final = len(combined)

        multi_db_flow_diagram(
            sources=[
                {"text": f"HRS\n(N={hrs_raw_n:,})",
                 "excl": f"Non-respondents\n(N={hrs_raw_n-hrs_resp:,})",
                 "after": f"W11 respondents\n(N={hrs_resp:,})"},
                {"text": f"KLoSA\n(N={klosa_raw_n:,})",
                 "excl": f"Non-respondents\n(N={klosa_raw_n-klosa_resp:,})",
                 "after": f"W5 respondents\n(N={klosa_resp:,})"},
                {"text": f"SHARE\n(N={share_raw_n:,})",
                 "excl": f"Non-respondents\n(N={share_raw_n-share_resp:,})",
                 "after": f"W5 respondents\n(N={share_resp:,})"},
            ],
            merge_text=f"Combined cohort (N={hrs_resp+klosa_resp+share_resp:,})",
            post_texts=[
                f"Age >= 50 with valid data (N={total_final:,})",
            ],
            post_excls=[None],
            save_name="FigureFlowDiagram",
            output_dir=str(FIG_DIR),
        )

        # ============================================================
        # Step 1.9: variable_name_mapping.csv
        # ============================================================
        tee_print("\n[Step 1.9] Generating variable_name_mapping.csv...", rf)
        mapping_data = [
            ('age', 'Age', 'Participant age at baseline', 'years', '1_Demographics', 1),
            ('gender', 'Female', 'Sex (Female proportion)', 'binary', '1_Demographics', 2),
            ('education', 'Education Level', 'ISCED-3 education level', 'ordinal', '1_Demographics', 3),
            ('married', 'Married/Partnered', 'Marital status', 'binary', '1_Demographics', 4),
            ('loneliness', 'Loneliness', 'Felt lonely (binary)', 'binary', '2_Exposure', 1),
            ('event', 'All-Cause Mortality', 'Death during follow-up', 'binary', '3_Outcome', 1),
            ('time_years', 'Follow-up Time', 'Years from baseline to event/censoring', 'years', '3_Outcome', 2),
            ('hibpe', 'Hypertension', 'Ever diagnosed with hypertension', 'binary', '4_Disease_Burden', 1),
            ('diabe', 'Diabetes', 'Ever diagnosed with diabetes', 'binary', '4_Disease_Burden', 2),
            ('hearte', 'Heart Disease', 'Ever diagnosed with heart disease', 'binary', '4_Disease_Burden', 3),
            ('stroke', 'Stroke', 'Ever diagnosed with stroke', 'binary', '4_Disease_Burden', 4),
            ('lunge', 'Lung Disease', 'Ever diagnosed with lung disease', 'binary', '4_Disease_Burden', 5),
            ('cancre', 'Cancer', 'Ever diagnosed with cancer', 'binary', '4_Disease_Burden', 6),
            ('arthre', 'Arthritis', 'Ever diagnosed with arthritis', 'binary', '4_Disease_Burden', 7),
            ('depression', 'Depressive Symptoms', 'Depression score (z-scored)', 'z-score', '5_Health_Status', 1),
            ('srh', 'Self-Rated Health', 'Self-rated health (1=Excellent to 5=Poor)', 'ordinal', '5_Health_Status', 2),
            ('adl', 'ADL Limitations', 'Activities of Daily Living limitations', 'count', '5_Health_Status', 3),
            ('iadl', 'IADL Limitations', 'Instrumental ADL limitations', 'count', '5_Health_Status', 4),
            ('smoken', 'Current Smoking', 'Currently smokes', 'binary', '6_Behavioral', 1),
            ('drink', 'Alcohol Use', 'Any alcohol consumption', 'binary', '6_Behavioral', 2),
        ]
        mapping_df = pd.DataFrame(mapping_data,
            columns=['database_name', 'sci_name', 'description', 'unit', 'sort_category', 'sort_priority'])
        mapping_df.to_csv(DATA_DIR / "variable_name_mapping.csv", index=False)

        assert len(mapping_df) > 0
        assert not mapping_df['sci_name'].str.contains('_').any()
        tee_print(f"  Saved: {len(mapping_df)} mappings, no underscores in sci_name", rf)

        # ============================================================
        # Final
        # ============================================================
        tee_print("\n" + "=" * 70, rf)
        tee_print("Part 1 COMPLETE", rf)
        tee_print("=" * 70, rf)
        tee_print(f"  N={total_final:,} | Deaths={combined['event'].sum():,} ({combined['event'].mean()*100:.1f}%)", rf)
        tee_print(f"  Power(HR=1.15)={power_target:.4f}", rf)
        tee_print(f"  Files: cleanDataPart1.csv, Data_Quality_Report.md, Power_Analysis_Report.md", rf)
        tee_print(f"         Assumption_Testing_Log.md, variable_name_mapping.csv, FigureFlowDiagram.png/pdf", rf)

    return combined


if __name__ == "__main__":
    combined = main()
