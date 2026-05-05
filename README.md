# Loneliness and All-Cause Mortality Among Older Adults: A Cross-National Inverse Probability Weighting Study

## Study Overview

This repository contains the reproducible analysis code for a cross-national causal inference study examining the effect of **loneliness on all-cause mortality** among community-dwelling older adults (age >= 50).

Using **Inverse Probability of Treatment Weighting (IPTW)** with weighted Cox proportional hazards regression, the study pools data from three longitudinal aging cohorts across three continents:

| Database | Country | Baseline Wave | Follow-up | Role |
|----------|---------|---------------|-----------|------|
| HRS | USA | W11 (2012) | ~10 years | Discovery |
| KLoSA | Korea | W5 (2014) | ~6 years | Replication |
| SHARE | Europe (27 countries) | W5 (2013) | ~7 years | Replication |

**Total sample**: ~92,839 participants | ~15,507 deaths

## Key Findings

- **Loneliness increases all-cause mortality risk by ~20%** after IPTW adjustment (HR = 1.20, P < 0.001)
- Cross-national consistency: significant effects in HRS (USA) and SHARE (Europe)
- Robust to 13+ sensitivity analyses (E-value, competing risks, alternate PS models, leave-one-database-out)
- Depression is the strongest driver of loneliness propensity (SHAP analysis)

## Repository Structure

```
.
├── code/                        # Analysis scripts (Python + R)
│   ├── Part1_Data_Cleaning/     # Data loading, cleaning, harmonization
│   ├── Part2_Feature_Engineering/ # IPTW confounders, baseline table
│   ├── Part3_Model_Development/  # Propensity score estimation, IPTW weights
│   ├── Part4_Model_Evaluation/   # Weighted Cox regression, survival curves
│   ├── Part5_Interpretability/   # SHAP analysis on PS model
│   ├── Part6_Sensitivity_Analysis/ # E-value, competing risks, LOO
│   └── Part7_Summary/            # Results summary, variable dictionary
├── tables/                      # Output tables (CSV, XLSX)
├── figures/                     # Publication-quality figures (PNG, PDF)
├── Supplement/                  # Supplementary materials
├── requirements.txt             # Python dependencies
├── CITATION.cff                 # Citation metadata (Zenodo DOI)
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Analysis Pipeline

The analysis is organized into 7 sequential parts:

| Part | Script | Description |
|------|--------|-------------|
| 1 | `part1_data_cleaning.py` | Load harmonized data from HRS, KLoSA, SHARE; filter age >= 50; harmonize variables; MICE imputation |
| 2 | `part2_feature_engineering.py` | Define IPTW confounders; VIF check; generate baseline characteristics table with SMD |
| 3 | `part3_model_development.py` | Fit PS models (Logistic Regression + Gradient Boosting); compute stabilized IPTW weights; Love plot |
| 4 | `part4_model_evaluation.py` + `part4_cox_analysis.R` | Weighted Cox PH regression via R `survival::coxph`; per-database + random-effects meta-analysis; Kaplan-Meier curves; forest plot |
| 5 | `part5_shap_analysis.py` | SHAP interpretability on PS model; feature importance ranking |
| 6 | `part6_sensitivity_analysis.py` + `part6_sensitivity.R` | E-value, alternate PS, trimming sensitivity, leave-one-DB-out, competing risks (Fine-Gray), PS matching |
| 7 | `part7_summary.py` | Generate results summary and variable dictionary |

## Data Access

This study uses publicly available longitudinal aging data. Due to data use agreements, **raw data is not included** in this repository. Users must register and download data independently from the original sources listed below.

### Original Data Sources

All three databases are publicly accessible to registered researchers:

| Database | Full Name | Provider | Website |
|----------|-----------|----------|---------|
| HRS | Health and Retirement Study | University of Michigan | [hrs.isr.umich.edu](https://hrs.isr.umich.edu/) |
| KLoSA | Korean Longitudinal Study of Ageing | Korea Employment Information Service (KEIS) | [survey.keis.or.kr](https://survey.keis.or.kr/) |
| SHARE | Survey of Health, Ageing and Retirement in Europe | SHARE-ERIC | [share-eric.eu](https://share-eric.eu/) |

### Data Preparation Workflow

The data preparation involves three steps:

1. **Download raw data**: Register at the original data sources above and download the raw public-use datasets (e.g., HRS public use files, KLoSA public data, SHARE release data).

2. **Variable harmonization via Gateway to Global Aging Data (g2aging)**: Use the [Gateway to Global Aging Data](https://g2aging.org/) (produced by the USC Center for Economic and Social Research) as a reference for variable harmonization. The g2aging project provides consistently coded variables across all participating longitudinal aging studies, enabling cross-national comparative research. Register at [g2aging.org](https://g2aging.org/) to access the harmonized variable definitions and codebooks, then harmonize the raw data accordingly.

3. **Convert to parquet format**: Save the harmonized datasets as `.parquet` files and place them in the `Original_data/` directory as described below.

For details on variable harmonization, refer to the [g2aging documentation](https://g2aging.org/documents) and codebooks.

## Environment Setup

### Python (>= 3.10)

```bash
pip install -r requirements.txt
```

### R (>= 4.2.0)

Install required R packages:

```r
install.packages(c("survival", "meta", "cmprsk"))
```

## Usage

### Step 1: Prepare data

After downloading the raw data from the original sources and harmonizing variables according to the g2aging codebooks, convert the harmonized datasets to parquet format and place them in the `Original_data/` directory at the project root:

```
Original_data/
├── HRS/
│   └── H_HRS.parquet          # Harmonized HRS (raw data from hrs.isr.umich.edu, harmonized via g2aging)
├── KLoSA/
│   └── H_KLoSA_e2.parquet     # Harmonized KLoSA (raw data from survey.keis.or.kr, harmonized via g2aging)
└── SHARE/
    └── H_SHARE_f2.parquet     # Harmonized SHARE (raw data from share-eric.eu, harmonized via g2aging)
```

> **Note**: If your harmonized data is in CSV or other formats, modify the `pd.read_parquet()` calls in `part1_data_cleaning.py` accordingly (e.g., to `pd.read_csv()`), or convert to parquet first.

### Step 2: Run analysis sequentially

```bash
python code/Part1_Data_Cleaning/part1_data_cleaning.py
python code/Part2_Feature_Engineering/part2_feature_engineering.py
python code/Part3_Model_Development/part3_model_development.py
python code/Part4_Model_Evaluation/part4_model_evaluation.py
python code/Part5_Interpretability/part5_shap_analysis.py
python code/Part6_Sensitivity_Analysis/part6_sensitivity_analysis.py
python code/Part7_Summary/part7_summary.py
```

Parts 4 and 6 call R scripts internally via `subprocess`, requiring R to be available in the system PATH.

## Output Files

### Main Tables
- `TableBaseline.xlsx` — Baseline characteristics by loneliness status (with SMD)
- `TablePrimaryResults.xlsx` — Primary and per-database Cox regression results
- `TableSMDBalance.csv` — Standardized mean differences before/after IPTW
- `TableFeatureImportance.xlsx` — SHAP-based feature importance
- `TableSensitivityResults.xlsx` — All sensitivity analysis results
- `TablePSModelComparison.csv` — PS model performance comparison

### Main Figures
- `FigureFlowDiagram.png` — Study flow diagram
- `FigureLovePlot.png` — Covariate balance (Love plot)
- `FigureSurvivalCurves.png` — Weighted Kaplan-Meier survival curves
- `FigureForestPlot.png` — Per-database forest plot
- `FigureSHAPSummary.png` — SHAP beeswarm summary plot
- `FigureSensitivityForest.png` — Sensitivity analyses forest plot

## Statistical Methods

### Causal Inference Framework
- **Propensity score**: Gradient Boosting classifier (selected over Logistic Regression based on AUC)
- **IPTW weights**: Stabilized weights with 1st/99th percentile trimming
- **Primary analysis**: Weighted Cox PH model with robust standard errors
- **Meta-analysis**: Random-effects model (DerSimonian-Laird) across databases

### Sensitivity Analyses
- E-value for unmeasured confounding
- Alternate PS model (Logistic Regression)
- Weight trimming sensitivity (no trim, 5/95, 1/99)
- Leave-one-database-out analysis
- Competing risks (Fine-Gray subdistribution model)
- PS matching (1:1 nearest-neighbor)
- Multivariable Cox regression (no IPTW)
- Time-varying effect test

## Citation

If you use this code, please cite:

```bibtex
@software{loneliness_mortality_iptw_2026,
  title = {Loneliness and All-Cause Mortality Among Older Adults: A Cross-National Inverse Probability Weighting Study},
  author = {},
  year = {2026},
  url = {https://github.com/<username>/loneliness-mortality-iptw},
  doi = {}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Data Use and Acknowledgments

The analysis uses publicly available survey data. Users must comply with the respective data use agreements:

- **HRS**: Produced and distributed by the University of Michigan with funding from the National Institute on Aging (NIA U01AG009740). Users must register at [hrs.isr.umich.edu](https://hrs.isr.umich.edu/) and agree to the HRS data use terms.
- **KLoSA**: Produced by the Korea Employment Information Service (KEIS). Data is available at [survey.keis.or.kr](https://survey.keis.or.kr/) upon registration.
- **SHARE**: Produced by SHARE-ERIC with funding from the European Commission. Data is available at [share-eric.eu](https://share-eric.eu/) upon registration.
- **Gateway to Global Aging Data**: The harmonized datasets were obtained from [g2aging.org](https://g2aging.org/), produced by the USC Center for Economic and Social Research (CESR) with funding from NIA (R01AG030153).
