# Supplement Index: T019 Loneliness → Mortality (IPTW)

## eFigures

| Supplement Figure | File | Description |
|-------------------|------|-------------|
| eFigure S1 | FigurePSDistribution.png/pdf | Propensity score distribution by treatment group (Gradient Boosting, AUC=0.870) |
| eFigure S2 | FigureIPTWWeights.png/pdf | IPTW weight distribution (unstabilized vs stabilized, with trimming) |
| eFigure S3 | FigureSHAPSummary.png/pdf | SHAP beeswarm plot for propensity score model (drivers of loneliness) |
| eFigure S4 | FigureFeatureImportance.png/pdf | Feature importance bar chart (mean |SHAP| values) |
| eFigure S5 | FigureSHAPDependence.png/pdf | SHAP dependence plots for top 6 features |
| eFigure S6 | FigureSensitivityForest.png/pdf | Forest plot of all 13 sensitivity analyses |

## eTables

| Supplement Table | File | Description |
|------------------|------|-------------|
| eTable S1 | TableSMDBalance.csv | Standardized mean differences before/after IPTW weighting |
| eTable S2 | TablePerDBSMDBalance.csv | Per-database SMD balance diagnostics |
| eTable S3 | TablePSModelComparison.csv | PS model comparison (Logistic Regression vs Gradient Boosting) |
| eTable S4 | TablePerDBPS.csv | Per-database PS model performance and effective sample sizes |
| eTable S5 | TableSensitivityResults.csv | All 13 sensitivity analyses with HR, 95% CI, P-values |
| eTable S6 | TableFeatureImportance.csv | SHAP-based feature importance for PS model |
| eTable S7 | TableEValue.csv | E-value calculation for unmeasured confounding |
| eTable S8 | TableVIF.csv | Variance inflation factor diagnostics for confounders |

## Main Text Figures

| Figure | File | Description |
|--------|------|-------------|
| Figure 1 | FigureFlowDiagram.png/pdf | Study flow diagram (N=92,839) |
| Figure 2 | FigureLovePlot.png/pdf | Covariate balance Love plot (SMD before/after IPTW) |
| Figure 3 | FigureSurvivalCurves.png/pdf | IPTW-weighted Kaplan-Meier survival curves |
| Figure 4 | FigureForestPlot.png/pdf | Per-database forest plot with random-effects meta-analysis |

## Main Text Tables

| Table | File | Description |
|-------|------|-------------|
| Table 1 | TableBaseline.xlsx | Baseline characteristics by loneliness status |
| Table 2 | TablePrimaryResults.xlsx | Primary and per-database IPTW-weighted Cox regression results |
