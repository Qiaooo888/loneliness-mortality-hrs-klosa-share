# ============================================================
# Part 3: Model Development — T019 Loneliness → Mortality (IPTW)
# Propensity Score estimation, IPTW weights, SMD balance, Love plot
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
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OTHER_DIR = OUTPUT_DIR / "intermediate"
TABLE_DIR = PROJECT_ROOT / "tables"
FIG_DIR = PROJECT_ROOT / "figures"

for d in [FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "Part3_result.txt"


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


def auc_bootstrap_ci(y_true, y_score, n=500, seed=42):
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def main():
    with open(RESULT_FILE, 'w', encoding='utf-8') as rf:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.__stdout__, rf)
        try:
            print("=" * 70)
            print("Part 3: Model Development — T019 IPTW (Propensity Score)")
            print("=" * 70)

            # ============================================================
            # Step 3.1: Load data
            # ============================================================
            print("\n[Step 3.1] Loading data...")
            df = pd.read_csv(OTHER_DIR / "cleanDataPart2.csv")
            mapping_df = pd.read_csv(DATA_DIR / "variable_name_mapping.csv")
            name_map = dict(zip(mapping_df['database_name'], mapping_df['sci_name']))
            confounder_list = pd.read_csv(DATA_DIR / "confounder_list.csv")['confounder'].tolist()
            print(f"  N={len(df):,}")
            print(f"  Confounders: {len(confounder_list)}")

            # ============================================================
            # Step 3.2: Data leakage check (IPTW-adapted)
            # ============================================================
            print("\n[Step 3.2] Data leakage check (IPTW-adapted)...")
            treatment = 'loneliness'
            outcome_vars = ['event', 'time_years']

            for ov in outcome_vars:
                if ov in confounder_list:
                    print(f"  [LEAK-1] {ov} in confounders! Removing.")
                    confounder_list.remove(ov)
            if treatment in confounder_list:
                print(f"  [LEAK-1] {treatment} in confounders! Removing.")
                confounder_list.remove(treatment)

            id_keywords = ['id', '_id', 'pid', 'uid', 'respondent', 'hhid']
            for feat in confounder_list[:]:
                for kw in id_keywords:
                    if kw in feat.lower():
                        print(f"  [LEAK-3] {feat} is ID column! Removing.")
                        confounder_list.remove(feat)
                        break

            for feat in confounder_list[:]:
                if df[feat].dtype in ['float64', 'int64', 'float32', 'int32']:
                    corr = abs(df[feat].corr(df[outcome_vars[0]]))
                    if corr > 0.95:
                        print(f"  [LEAK-4] {feat} corr={corr:.4f}! Removing.")
                        confounder_list.remove(feat)

            print(f"  Confounders after leak check: {len(confounder_list)}")
            print("  PASSED: No leakage detected")

            # ============================================================
            # Step 3.2a: VIF check on confounders
            # ============================================================
            print("\n[Step 3.2a] VIF check on confounders...")
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            X_vif = df[confounder_list].dropna()
            vif_data = []
            for i, col in enumerate(confounder_list):
                vif = variance_inflation_factor(X_vif.values, i)
                vif_data.append({'feature': col, 'VIF': vif})
                if vif > 10:
                    vif_str = "inf" if np.isinf(vif) else f"{vif:.1f}"
                    print(f"  {col}: VIF={vif_str} (>10)")

            vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
            vif_df.to_csv(TABLE_DIR / "TableVIF.csv", index=False)
            print(f"  Saved: Table/TableVIF.csv")
            print(f"  Max VIF={vif_df['VIF'].replace([np.inf], 999).max():.2f}")

            # ============================================================
            # Step 3.3: Prepare features
            # ============================================================
            print("\n[Step 3.3] Preparing features...")
            X = df[confounder_list].copy()
            y = df[treatment].values

            missing = X.isnull().sum()
            if missing.sum() > 0:
                print(f"  WARNING: Missing values:")
                for col in missing[missing > 0].index:
                    print(f"    {col}: {missing[col]}")
            else:
                print(f"  No missing values")

            print(f"  X shape: {X.shape}")
            print(f"  Treatment: {y.sum():,.0f}/{len(y):,} ({y.mean()*100:.1f}%)")

            # ============================================================
            # Step 3.4: Fit Logistic Regression PS model
            # ============================================================
            print("\n[Step 3.4] Fitting PS models...")

            lr_model = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=42)
            lr_model.fit(X, y)
            ps_lr = lr_model.predict_proba(X)[:, 1]

            auc_lr = roc_auc_score(y, ps_lr)
            brier_lr = brier_score_loss(y, ps_lr)
            logloss_lr = log_loss(y, ps_lr)

            gb_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=50, subsample=0.8, random_state=42
            )
            gb_model.fit(X, y)
            ps_gb = gb_model.predict_proba(X)[:, 1]

            auc_gb = roc_auc_score(y, ps_gb)
            brier_gb = brier_score_loss(y, ps_gb)
            logloss_gb = log_loss(y, ps_gb)

            # Bootstrap 95% CI
            print(f"  Computing AUC 95% CI (500 bootstrap)...")
            ci_lr = auc_bootstrap_ci(y, ps_lr)
            ci_gb = auc_bootstrap_ci(y, ps_gb)

            print(f"  Logistic Regression:")
            print(f"    AUC={auc_lr:.4f} (95% CI: {ci_lr[0]:.4f}-{ci_lr[1]:.4f})")
            print(f"    Brier={brier_lr:.4f}, LogLoss={logloss_lr:.4f}")
            print(f"  Gradient Boosting:")
            print(f"    AUC={auc_gb:.4f} (95% CI: {ci_gb[0]:.4f}-{ci_gb[1]:.4f})")
            print(f"    Brier={brier_gb:.4f}, LogLoss={logloss_gb:.4f}")

            # Select best
            if auc_gb > auc_lr:
                ps_model = gb_model
                ps_scores = ps_gb.copy()
                ps_model_name = "GradientBoosting"
                selected_auc = auc_gb
                selected_ci = ci_gb
            else:
                ps_model = lr_model
                ps_scores = ps_lr.copy()
                ps_model_name = "LogisticRegression"
                selected_auc = auc_lr
                selected_ci = ci_lr
            print(f"  Selected: {ps_model_name} (AUC={selected_auc:.4f})")

            # AUC sanity check
            if selected_auc >= 0.99:
                raise ValueError(f"AUC={selected_auc:.4f} >= 0.99, leakage suspected!")
            elif selected_auc >= 0.95:
                raise ValueError(f"AUC={selected_auc:.4f} >= 0.95, severe leakage suspected!")
            elif selected_auc >= 0.90:
                print(f"  WARNING: AUC >= 0.90, review for potential leakage")
            elif selected_auc < 0.60:
                print(f"  WARNING: AUC < 0.60, PS model may be too weak")
            else:
                print(f"  AUC in reasonable range")

            # Save PS model comparison table
            ps_comp = pd.DataFrame([
                {'Model': 'Logistic Regression', 'AUC': auc_lr,
                 'AUC_95CI_Lower': ci_lr[0], 'AUC_95CI_Upper': ci_lr[1],
                 'Brier': brier_lr, 'LogLoss': logloss_lr},
                {'Model': 'Gradient Boosting', 'AUC': auc_gb,
                 'AUC_95CI_Lower': ci_gb[0], 'AUC_95CI_Upper': ci_gb[1],
                 'Brier': brier_gb, 'LogLoss': logloss_gb},
            ])
            ps_comp.to_csv(TABLE_DIR / "TablePSModelComparison.csv", index=False)
            print(f"  Saved: Table/TablePSModelComparison.csv")

            # ============================================================
            # Step 3.5: Compute IPTW weights (with percentile trimming)
            # ============================================================
            print("\n[Step 3.5] Computing IPTW weights...")
            ps = ps_scores.copy()
            ps = np.clip(ps, 0.001, 0.999)

            p_treatment = y.mean()
            iptw_raw = np.where(y == 1, p_treatment / ps, (1 - p_treatment) / (1 - ps))

            # Weight-based 1st/99th percentile trimming
            lo, hi = np.percentile(iptw_raw, [1, 99])
            iptw_stab = np.clip(iptw_raw, lo, hi)
            print(f"  Weight trimming: 1st pct={lo:.4f}, 99th pct={hi:.4f}")

            print(f"  Stabilized weights (trimmed):")
            print(f"    Mean={iptw_stab.mean():.4f}, Median={np.median(iptw_stab):.4f}")
            print(f"    Min={iptw_stab.min():.4f}, Max={iptw_stab.max():.4f}")
            print(f"    % > 5: {(iptw_stab > 5).mean()*100:.1f}%")

            df['ps_score'] = ps_scores
            df['iptw'] = iptw_stab

            # ============================================================
            # Step 3.5a: Effective Sample Size (ESS)
            # ============================================================
            print("\n[Step 3.5a] Effective Sample Size (ESS)...")
            ess_total = (iptw_stab.sum() ** 2) / (iptw_stab ** 2).sum()
            ess_treat = (iptw_stab[y == 1].sum() ** 2) / (iptw_stab[y == 1] ** 2).sum()
            ess_ctrl = (iptw_stab[y == 0].sum() ** 2) / (iptw_stab[y == 0] ** 2).sum()
            print(f"  ESS total:     {ess_total:,.0f} (orig {len(y):,})")
            print(f"  ESS treated:   {ess_treat:,.0f} (orig {y.sum():,.0f})")
            print(f"  ESS control:   {ess_ctrl:,.0f} (orig {(1-y).sum():,.0f})")
            print(f"  ESS ratio:     {ess_total/len(y)*100:.1f}%")

            # ============================================================
            # Step 3.5b: SMD balance diagnostics + Love plot
            # ============================================================
            print("\n[Step 3.5b] SMD balance diagnostics...")
            smd_rows = []
            for feat in confounder_list:
                smd_pre = compute_smd(X[feat].values, y, weights=None)
                smd_post = compute_smd(X[feat].values, y, weights=iptw_stab)
                sci = name_map.get(feat, feat)
                smd_rows.append({
                    'covariate': feat,
                    'sci_name': sci,
                    'SMD_unweighted': round(smd_pre, 4),
                    'SMD_weighted': round(smd_post, 4),
                    'abs_SMD_weighted': round(abs(smd_post), 4),
                    'balanced': 'Yes' if abs(smd_post) < 0.1 else 'No'
                })

            smd_df = pd.DataFrame(smd_rows).sort_values('abs_SMD_weighted', ascending=False)
            smd_df.to_csv(TABLE_DIR / "TableSMDBalance.csv", index=False)

            n_unbal = (smd_df['abs_SMD_weighted'] >= 0.1).sum()
            print(f"  Saved: Table/TableSMDBalance.csv")
            print(f"  |SMD| >= 0.1 after weighting: {n_unbal}/{len(smd_df)}")
            print(f"\n  Top 5 by |SMD|:")
            for _, row in smd_df.head(5).iterrows():
                print(f"    {row['sci_name']}: {row['SMD_unweighted']:.4f} -> {row['SMD_weighted']:.4f} ({row['balanced']})")

            # Love plot
            smd_plot = smd_df.sort_values('SMD_unweighted', key=abs, ascending=True)
            fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(smd_plot))))
            yy = np.arange(len(smd_plot))
            labels = [name_map.get(c, c) for c in smd_plot['covariate']]
            ax.scatter(smd_plot['SMD_unweighted'].abs(), yy, label='Before IPTW', color='#4C72B0', s=40, zorder=3)
            ax.scatter(smd_plot['SMD_weighted'].abs(), yy, label='After IPTW', color='#DD8452', s=40, zorder=3)
            ax.axvline(0.1, ls='--', color='red', alpha=0.7, label='SMD = 0.1')
            ax.set_yticks(yy)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel('|Standardized Mean Difference|')
            ax.legend(fontsize=8)
            ax.set_title('Covariate Balance (Love Plot)')
            plt.tight_layout()
            fig.savefig(FIG_DIR / "FigureLovePlot.png", dpi=300)
            fig.savefig(FIG_DIR / "FigureLovePlot.pdf")
            plt.close()
            print(f"  Saved: FigureLovePlot.png/pdf")

            # ============================================================
            # Step 3.6: PS distribution figures
            # ============================================================
            print("\n[Step 3.6] PS distribution figures...")
            lonely_ps = ps[y == 1]
            not_lonely_ps = ps[y == 0]
            print(f"  Lonely PS:     mean={lonely_ps.mean():.4f}, range=[{lonely_ps.min():.4f}, {lonely_ps.max():.4f}]")
            print(f"  Not Lonely PS: mean={not_lonely_ps.mean():.4f}, range=[{not_lonely_ps.min():.4f}, {not_lonely_ps.max():.4f}]")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(not_lonely_ps, bins=50, alpha=0.6, density=True, label='Not Lonely', color='#4C72B0')
            ax.hist(lonely_ps, bins=50, alpha=0.6, density=True, label='Lonely', color='#DD8452')
            ax.set_xlabel('Propensity Score')
            ax.set_ylabel('Density')
            ax.set_title(f'PS Distribution ({ps_model_name}, AUC={selected_auc:.3f})')
            ax.legend()
            plt.tight_layout()
            fig.savefig(FIG_DIR / "FigurePSDistribution.png", dpi=300)
            fig.savefig(FIG_DIR / "FigurePSDistribution.pdf")
            plt.close()
            print(f"  Saved: FigurePSDistribution.png/pdf")

            # Weight distribution
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(iptw_raw, bins=100, alpha=0.7, color='#55A868')
            axes[0].set_xlabel('IPTW Weight (before trimming)')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Raw Stabilized Weights')

            axes[1].hist(iptw_stab, bins=100, alpha=0.7, color='#C44E52')
            axes[1].set_xlabel('IPTW Weight (after trimming)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Trimmed Stabilized Weights')
            plt.tight_layout()
            fig.savefig(FIG_DIR / "FigureIPTWWeights.png", dpi=300)
            fig.savefig(FIG_DIR / "FigureIPTWWeights.pdf")
            plt.close()
            print(f"  Saved: FigureIPTWWeights.png/pdf")

            # ============================================================
            # Step 3.7: Per-database PS model
            # ============================================================
            print("\n[Step 3.7] Per-database PS estimation...")
            per_db_results = []
            for db_name in ['HRS', 'KLoSA', 'SHARE']:
                db_mask = df['database'] == db_name
                db_df = df[db_mask]
                db_X = db_df[confounder_list]
                db_y = db_df[treatment].values
                if len(db_df) < 100:
                    print(f"  {db_name}: N={len(db_df)} too small, skipping")
                    continue

                if ps_model_name == "GradientBoosting":
                    db_model = GradientBoostingClassifier(
                        n_estimators=200, max_depth=4, learning_rate=0.1,
                        min_samples_leaf=50, subsample=0.8, random_state=42
                    )
                else:
                    db_model = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=42)
                db_model.fit(db_X, db_y)
                db_ps = db_model.predict_proba(db_X)[:, 1]
                db_ps = np.clip(db_ps, 0.001, 0.999)

                db_auc = roc_auc_score(db_y, db_ps)
                db_p_treat = db_y.mean()
                db_iptw_raw = np.where(db_y == 1, db_p_treat / db_ps, (1 - db_p_treat) / (1 - db_ps))
                db_lo, db_hi = np.percentile(db_iptw_raw, [1, 99])
                db_iptw = np.clip(db_iptw_raw, db_lo, db_hi)

                db_ess = (db_iptw.sum() ** 2) / (db_iptw ** 2).sum()

                df.loc[db_mask, f'ps_{db_name}'] = db_ps
                df.loc[db_mask, f'iptw_{db_name}'] = db_iptw

                per_db_results.append({
                    'Database': db_name, 'N': len(db_df),
                    'Lonely': int(db_y.sum()), 'Lonely_pct': round(db_y.mean() * 100, 1),
                    'AUC': round(db_auc, 4), 'ESS': int(db_ess),
                    'IPTW_mean': round(db_iptw.mean(), 4)
                })
                print(f"  {db_name}: N={len(db_df):,}, Lonely={db_y.sum():,.0f} ({db_y.mean()*100:.1f}%), "
                      f"AUC={db_auc:.4f}, ESS={db_ess:,.0f}, IPTW mean={db_iptw.mean():.4f}")

            per_db_df = pd.DataFrame(per_db_results)
            per_db_df.to_csv(TABLE_DIR / "TablePerDBPS.csv", index=False)
            print(f"  Saved: Table/TablePerDBPS.csv")

            # ============================================================
            # Step 3.8: Save model
            # ============================================================
            print("\n[Step 3.8] Saving model...")
            model_path = OTHER_DIR / "model_primary.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': ps_model,
                    'features': confounder_list,
                    'model_name': ps_model_name,
                    'metadata': {
                        'part': 'Part3',
                        'date': '2026-05-01',
                        'model_type': 'PropensityScore',
                        'study_type': 'IPTW',
                        'n_confounders': len(confounder_list),
                        'auc_lr': auc_lr,
                        'auc_gb': auc_gb,
                        'selected_model': ps_model_name,
                        'ess_total': ess_total,
                        'ess_treat': ess_treat,
                        'ess_ctrl': ess_ctrl,
                        'n_unbalanced_smd': n_unbal,
                        'weight_trim_pct': [1, 99]
                    }
                }, f)
            print(f"  Saved: model_primary.pkl")

            # ============================================================
            # Step 3.9: Save updated data
            # ============================================================
            df.to_csv(OTHER_DIR / "cleanDataPart3.csv", index=False)
            print(f"  Saved: cleanDataPart3.csv ({len(df):,} rows, {len(df.columns)} cols)")

            # ============================================================
            # Step 3.10: Summary
            # ============================================================
            print("\n" + "=" * 70)
            print("Part 3 COMPLETE")
            print("=" * 70)
            print(f"  PS model: {ps_model_name}")
            print(f"  AUC: {selected_auc:.4f} (95% CI: {selected_ci[0]:.4f}-{selected_ci[1]:.4f})")
            print(f"  LR AUC: {auc_lr:.4f}, GB AUC: {auc_gb:.4f}")
            print(f"  Confounders: {len(confounder_list)}")
            print(f"  |SMD| >= 0.1 after IPTW: {n_unbal}/{len(smd_df)}")
            print(f"  ESS: {ess_total:,.0f} / {len(y):,} ({ess_total/len(y)*100:.1f}%)")
            print(f"  Weight trimming: 1st/99th percentile")
            print(f"  Per-DB: HRS, KLoSA, SHARE")
            print(f"  Tables: TableSMDBalance, TablePSModelComparison, TablePerDBPS, TableVIF")
            print(f"  Figures: FigureLovePlot, FigurePSDistribution, FigureIPTWWeights")

        finally:
            sys.stdout = original_stdout

    return df


if __name__ == "__main__":
    df = main()
