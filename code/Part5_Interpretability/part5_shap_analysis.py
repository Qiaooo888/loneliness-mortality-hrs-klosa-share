# ============================================================
# Part 5: SHAP Interpretability — T019 Loneliness → Mortality (IPTW)
# SHAP analysis on PS model to understand treatment selection drivers
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
import shap
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

RESULT_FILE = OUTPUT_DIR / "Part5_result.txt"


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
            print("Part 5: SHAP Interpretability — T019 IPTW (PS Model)")
            print("=" * 70)

            # ============================================================
            # Step 5.1: Load model and data
            # ============================================================
            print("\n[Step 5.1] Loading model and data...")
            with open(OTHER_DIR / "model_primary.pkl", 'rb') as f:
                model_data = pickle.load(f)
            ps_model = model_data['model']
            feature_names = model_data['features']
            model_name = model_data['model_name']

            df = pd.read_csv(OTHER_DIR / "cleanDataPart3.csv")
            print(f"  Model: {model_name}")
            print(f"  Features: {len(feature_names)}")
            print(f"  N={len(df):,}")

            X = df[feature_names].copy()

            # ================================================================
            # ★★★ Step 5.1a: SCI Variable Name Mapping (MANDATORY) ★★★
            # ================================================================
            print("\n[Step 5.1a] SCI Variable Name Mapping...")
            mapping_df = pd.read_csv(DATA_DIR / "variable_name_mapping.csv")
            name_map = dict(zip(mapping_df['database_name'], mapping_df['sci_name']))

            X_display = X.copy()
            X_display.columns = [name_map.get(col, col) for col in X_display.columns]

            display_names = [name_map.get(f, f) for f in feature_names]
            print(f"  Loaded {len(name_map)} mappings")
            for i, (db, sci) in enumerate(zip(feature_names, display_names)):
                print(f"    {db} -> {sci}")
                if i >= 4:
                    print(f"    ... ({len(feature_names) - 5} more)")
                    break

            # ============================================================
            # Step 5.2: Compute SHAP values
            # ============================================================
            print("\n[Step 5.2] Computing SHAP values...")
            if model_name == "GradientBoosting":
                explainer = shap.TreeExplainer(ps_model)
                # Sample for efficiency
                sample_size = min(10000, len(X))
                np.random.seed(42)
                sample_idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_idx]
                X_display_sample = X_display.iloc[sample_idx]

                shap_values = explainer.shap_values(X_sample)
                print(f"  TreeExplainer: {sample_size:,} samples")
            else:
                # Logistic Regression: use LinearExplainer or KernelExplainer
                background = shap.sample(X, 100, random_state=42)
                explainer = shap.KernelExplainer(ps_model.predict_proba, background)
                sample_size = min(5000, len(X))
                np.random.seed(42)
                sample_idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_idx]
                X_display_sample = X_display.iloc[sample_idx]
                shap_values = explainer.shap_values(X_sample)[:, 1]
                print(f"  KernelExplainer: {sample_size:,} samples")

            print(f"  SHAP values shape: {np.array(shap_values).shape}")

            # ============================================================
            # Step 5.3: SHAP Summary Plot (beeswarm)
            # ============================================================
            print("\n[Step 5.3] SHAP Summary Plot...")
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, X_display_sample, show=False, max_display=17)
            plt.tight_layout()
            fig.savefig(FIG_DIR / "FigureSHAPSummary.png", dpi=300, bbox_inches='tight')
            fig.savefig(FIG_DIR / "FigureSHAPSummary.pdf", bbox_inches='tight')
            plt.close()
            print(f"  Saved: FigureSHAPSummary.png/pdf")

            # ============================================================
            # Step 5.4: SHAP Bar Plot (mean |SHAP|)
            # ============================================================
            print("\n[Step 5.4] SHAP Feature Importance (Bar Plot)...")
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'database_name': feature_names,
                'sci_name': display_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

            # Sort by importance (descending, highest at top)
            importance_df_sorted = importance_df.sort_values('mean_abs_shap', ascending=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importance_df_sorted)))
            ax.barh(range(len(importance_df_sorted)), importance_df_sorted['mean_abs_shap'].values,
                    color=colors)
            ax.set_yticks(range(len(importance_df_sorted)))
            ax.set_yticklabels(importance_df_sorted['sci_name'].values, fontsize=9)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Feature Importance for Propensity of Loneliness')
            plt.tight_layout()
            fig.savefig(FIG_DIR / "FigureFeatureImportance.png", dpi=300, bbox_inches='tight')
            fig.savefig(FIG_DIR / "FigureFeatureImportance.pdf", bbox_inches='tight')
            plt.close()
            print(f"  Saved: FigureFeatureImportance.png/pdf")

            # ============================================================
            # Step 5.5: SHAP Dependence Plots (top 6 features)
            # ============================================================
            print("\n[Step 5.5] SHAP Dependence Plots (top 6)...")
            top6 = importance_df.head(6)['database_name'].tolist()
            top6_sci = importance_df.head(6)['sci_name'].tolist()

            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes = axes.flatten()
            for i, (feat, sci) in enumerate(zip(top6, top6_sci)):
                feat_idx = feature_names.index(feat)
                feat_vals = X_sample[feat].values
                feat_shap = shap_values[:, feat_idx]

                ax = axes[i]
                sc = ax.scatter(feat_vals, feat_shap, alpha=0.3, s=5, c='#4C72B0')
                ax.set_xlabel(sci, fontsize=9)
                ax.set_ylabel(f'SHAP value for {sci}', fontsize=8)
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax.tick_params(labelsize=7)
            plt.tight_layout()
            fig.savefig(FIG_DIR / "FigureSHAPDependence.png", dpi=300, bbox_inches='tight')
            fig.savefig(FIG_DIR / "FigureSHAPDependence.pdf", bbox_inches='tight')
            plt.close()
            print(f"  Saved: FigureSHAPDependence.png/pdf")

            # ============================================================
            # Step 5.6: Feature Importance Table
            # ============================================================
            print("\n[Step 5.6] Feature Importance Table...")
            importance_df.to_csv(TABLE_DIR / "TableFeatureImportance.csv", index=False)
            importance_df.to_excel(TABLE_DIR / "TableFeatureImportance.xlsx", index=False)
            print(f"  Saved: Table/TableFeatureImportance.csv + .xlsx")
            print(f"\n  Top 10 features by importance:")
            for i, row in importance_df.head(10).iterrows():
                print(f"    {i+1}. {row['sci_name']}: {row['mean_abs_shap']:.4f}")

            # ================================================================
            # ★★★ Step 5.9a: Variable Name Mapping Verification (MANDATORY) ★★★
            # ================================================================
            print("\n[Step 5.9a] Variable Name Verification...")
            # Check no database_name in display
            violations = []
            for db_name in feature_names:
                # Check if database_name appears in figure labels
                for fig_file in ['FigureSHAPSummary.png', 'FigureFeatureImportance.png']:
                    if not (FIG_DIR / fig_file).exists():
                        violations.append(f"Missing figure: {fig_file}")

            # Check X_display columns are sci_names
            for col in X_display.columns:
                if col in feature_names and col != name_map.get(col, col):
                    violations.append(f"X_display has database_name: {col}")

            if violations:
                print("  VIOLATIONS DETECTED:")
                for v in violations:
                    print(f"    {v}")
            else:
                print("  PASSED: All figures use sci_name, no database_name in display")

            # ============================================================
            # Step 5.7: Summary
            # ============================================================
            print("\n" + "=" * 70)
            print("Part 5 COMPLETE")
            print("=" * 70)
            print(f"  Model: {model_name} PS")
            print(f"  SHAP samples: {len(X_sample):,}")
            print(f"  Top 3 drivers of loneliness:")
            for i, row in importance_df.head(3).iterrows():
                print(f"    {row['sci_name']} (|SHAP|={row['mean_abs_shap']:.4f})")
            print(f"  Tables: TableFeatureImportance")
            print(f"  Figures: FigureSHAPSummary, FigureFeatureImportance, FigureSHAPDependence")

        finally:
            sys.stdout = original_stdout

    return importance_df


if __name__ == "__main__":
    importance = main()
