# Part4: Weighted Cox Regression — T019 IPTW Study
# Uses survival::coxph with IPTW weights
# Called from Python via subprocess

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]

library(survival)
library(meta)

# P-value formatting
fmt_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.001) return("<0.001")
  sprintf("%.3f", p)
}

# Load data
df <- read.csv(input_file, stringsAsFactors = FALSE)
cat(paste0("Loaded: ", nrow(df), " rows\n"))

# Common column structure
make_row <- function(analysis, database, n, events, hr, ci_low, ci_high, se, p, ph_p,
                     i2=NA_real_, tau2=NA_real_, q=NA_real_, q_p=NA_real_,
                     log_hr_raw=NA_real_, se_raw=NA_real_) {
  data.frame(
    Analysis = analysis, Database = database, N = n, Events = events,
    HR = round(hr, 4), CI_lower = round(ci_low, 4), CI_upper = round(ci_high, 4),
    Robust_SE = round(se, 4), P_value = p, PH_P = ph_p,
    I2 = round(i2, 2), tau2 = round(tau2, 4), Q = round(q, 3), Q_P = q_p,
    log_hr_raw = log_hr_raw, se_raw = se_raw
  )
}

# ============================================================
# 1. Overall pooled analysis
# ============================================================
cat("\n=== OVERALL WEIGHTED COX REGRESSION ===\n")

cox_overall <- coxph(
  Surv(time_years, event) ~ loneliness,
  data = df,
  weights = iptw,
  robust = TRUE
)

s <- summary(cox_overall)
hr <- s$coefficients[1, "exp(coef)"]
coef_raw <- s$coefficients[1, "coef"]
se <- s$coefficients[1, "robust se"]
p <- s$coefficients[1, "Pr(>|z|)"]
ci_low <- s$conf.int[1, "lower .95"]
ci_high <- s$conf.int[1, "upper .95"]
n_events <- s$nevent

ph_test <- cox.zph(cox_overall, transform = "km")
ph_p <- ph_test$table[1, 3]

cat(paste0("  HR = ", round(hr, 4), " (95% CI: ", round(ci_low, 4), "-", round(ci_high, 4), ")\n"))
cat(paste0("  Robust SE = ", round(se, 4), "\n"))
cat(paste0("  P = ", fmt_p(p), "\n"))
cat(paste0("  Events = ", n_events, "\n"))
cat(paste0("  PH test P = ", fmt_p(ph_p), "\n"))

overall_result <- make_row(
  "Overall", "Pooled", nrow(df), n_events,
  hr, ci_low, ci_high, se, p, ph_p,
  log_hr_raw = coef_raw, se_raw = se
)

# ============================================================
# 2. Per-database analysis
# ============================================================
cat("\n=== PER-DATABASE WEIGHTED COX REGRESSION ===\n")

db_results <- list()
for (db in c("HRS", "KLoSA", "SHARE")) {
  db_df <- df[df$database == db, ]
  if (nrow(db_df) < 100) {
    cat(paste0("  ", db, ": too few observations, skipping\n"))
    next
  }

  wt_col <- paste0("iptw_", db)
  if (wt_col %in% names(db_df)) {
    weights <- db_df[[wt_col]]
  } else {
    weights <- db_df$iptw
  }

  cox_db <- coxph(
    Surv(time_years, event) ~ loneliness,
    data = db_df,
    weights = weights,
    robust = TRUE
  )

  s_db <- summary(cox_db)
  hr_db <- s_db$coefficients[1, "exp(coef)"]
  coef_db <- s_db$coefficients[1, "coef"]
  se_db <- s_db$coefficients[1, "robust se"]
  p_db <- s_db$coefficients[1, "Pr(>|z|)"]
  ci_low_db <- s_db$conf.int[1, "lower .95"]
  ci_high_db <- s_db$conf.int[1, "upper .95"]
  n_events_db <- s_db$nevent

  ph_db <- cox.zph(cox_db, transform = "km")
  ph_p_db <- ph_db$table[1, 3]

  cat(paste0("  ", db, ": HR=", round(hr_db, 4),
             " (", round(ci_low_db, 4), "-", round(ci_high_db, 4), ")",
             ", P=", fmt_p(p_db),
             ", Events=", n_events_db, "\n"))

  db_results[[db]] <- make_row(
    "Per-Database", db, nrow(db_df), n_events_db,
    hr_db, ci_low_db, ci_high_db, se_db, p_db, ph_p_db,
    log_hr_raw = coef_db, se_raw = se_db
  )
}

# ============================================================
# 3. Random-effects meta-analysis (using raw values)
# ============================================================
cat("\n=== RANDOM-EFFECTS META-ANALYSIS ===\n")

log_hr <- sapply(db_results, function(x) x$log_hr_raw)
log_se <- sapply(db_results, function(x) x$se_raw)
db_names <- sapply(db_results, function(x) x$Database)

meta_result <- metagen(
  TE = log_hr,
  seTE = log_se,
  studlab = db_names,
  sm = "HR",
  method.tau = "DL",
  method.tau.ci = "J",
  hakn = FALSE
)

meta_hr <- exp(meta_result$TE.fixed)
meta_ci_low <- exp(meta_result$lower.fixed)
meta_ci_high <- exp(meta_result$upper.fixed)
meta_p <- meta_result$pval.fixed

meta_re_hr <- exp(meta_result$TE.random)
meta_re_ci_low <- exp(meta_result$lower.random)
meta_re_ci_high <- exp(meta_result$upper.random)
meta_re_p <- meta_result$pval.random

i2_val <- meta_result$I2 * 100
tau2_val <- meta_result$tau2
q_val <- meta_result$Q
q_p <- meta_result$pval.Q

cat(paste0("  Fixed-effect: HR=", round(meta_hr, 4),
           " (", round(meta_ci_low, 4), "-", round(meta_ci_high, 4), ")",
           ", P=", fmt_p(meta_p), "\n"))
cat(paste0("  Random-effects (DL): HR=", round(meta_re_hr, 4),
           " (", round(meta_re_ci_low, 4), "-", round(meta_re_ci_high, 4), ")",
           ", P=", fmt_p(meta_re_p), "\n"))
cat(paste0("  Heterogeneity: I2=", round(i2_val, 1), "%",
           ", tau2=", round(tau2_val, 4),
           ", Q=", round(q_val, 2),
           ", Q_P=", fmt_p(q_p), "\n"))

meta_row <- make_row(
  "Meta-Analysis (RE)", "Random-Effects",
  sum(sapply(db_results, function(x) x$N)),
  sum(sapply(db_results, function(x) x$Events)),
  meta_re_hr, meta_re_ci_low, meta_re_ci_high, NA, meta_re_p, NA,
  i2 = i2_val, tau2 = tau2_val, q = q_val, q_p = q_p
)

# ============================================================
# 4. Combine and save
# ============================================================
cat("\n=== RESULTS SUMMARY ===\n")
forest_data <- do.call(rbind, c(
  list(overall_result),
  db_results,
  list(meta_row)
))

# Remove raw precision columns for output
output_cols <- c("Analysis", "Database", "N", "Events", "HR", "CI_lower", "CI_upper",
                 "Robust_SE", "P_value", "PH_P", "I2", "tau2", "Q", "Q_P")
write.csv(forest_data[, output_cols], output_file, row.names = FALSE)
cat(paste0("\nResults saved to: ", output_file, "\n"))

# Print summary table
for (i in 1:nrow(forest_data)) {
  r <- forest_data[i, ]
  cat(paste0("  ", r$Analysis, " | ", r$Database, " | HR=", r$HR,
             " (", r$CI_lower, "-", r$CI_upper, ") | P=", fmt_p(r$P_value),
             ifelse(is.na(r$I2), "", paste0(" | I2=", r$I2, "%")),
             "\n"))
}
