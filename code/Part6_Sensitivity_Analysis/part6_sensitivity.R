# Part6: Sensitivity Analyses — T019 IPTW Study
# Extended version with trimming, LOO, PS matching
# Called from Python via subprocess

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]

library(survival)
library(meta)

fmt_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.001) return("<0.001")
  sprintf("%.3f", p)
}

df <- read.csv(input_file, stringsAsFactors = FALSE)
cat(paste0("Loaded: ", nrow(df), " rows\n"))

all_results <- list()
idx <- 1

confounders <- c('age', 'gender', 'education', 'married',
                 'hibpe', 'diabe', 'hearte', 'stroke', 'lunge', 'cancre', 'arthre',
                 'depression', 'srh', 'adl', 'iadl', 'smoken', 'drink')

add_result <- function(label, hr, ci_lo, ci_hi, p) {
  all_results[[idx]] <<- data.frame(
    Analysis = label, HR = round(hr, 4),
    CI_lower = round(ci_lo, 4), CI_upper = round(ci_hi, 4), P = p
  )
  idx <<- idx + 1
  cat(paste0("  ", label, ": HR=", round(hr, 4),
             " (", round(ci_lo, 4), "-", round(ci_hi, 4), "), P=", fmt_p(p), "\n"))
}

# ============================================================
# 1. Primary (reference)
# ============================================================
cat("\n=== 1. PRIMARY (GB PS, IPTW) ===\n")
cox1 <- coxph(Surv(time_years, event) ~ loneliness, data = df,
              weights = iptw, robust = TRUE)
s1 <- summary(cox1)
add_result("1. Primary (GB PS IPTW)",
           s1$coefficients[1, "exp(coef)"],
           s1$conf.int[1, "lower .95"],
           s1$conf.int[1, "upper .95"],
           s1$coefficients[1, "Pr(>|z|)"])

# ============================================================
# 2. Alternate PS: Logistic Regression
# ============================================================
cat("\n=== 2. ALTERNATE PS (LOGISTIC REGRESSION) ===\n")
lr_ps <- glm(as.factor(loneliness) ~ ., data = df[, c('loneliness', confounders)],
             family = binomial(link = "logit"))
ps_lr <- predict(lr_ps, type = "response")
ps_lr <- pmin(pmax(ps_lr, 0.001), 0.999)

p_treat <- mean(df$loneliness)
iptw_lr_raw <- ifelse(df$loneliness == 1, p_treat / ps_lr, (1 - p_treat) / (1 - ps_lr))
lo <- quantile(iptw_lr_raw, 0.01)
hi <- quantile(iptw_lr_raw, 0.99)
iptw_lr <- pmin(pmax(iptw_lr_raw, lo), hi)

cox2 <- coxph(Surv(time_years, event) ~ loneliness, data = df,
              weights = iptw_lr, robust = TRUE)
s2 <- summary(cox2)
add_result("2. LR PS (Alternate)",
           s2$coefficients[1, "exp(coef)"],
           s2$conf.int[1, "lower .95"],
           s2$conf.int[1, "upper .95"],
           s2$coefficients[1, "Pr(>|z|)"])

# ============================================================
# 3. Crude (unweighted)
# ============================================================
cat("\n=== 3. CRUDE (UNWEIGHTED) ===\n")
cox3 <- coxph(Surv(time_years, event) ~ loneliness, data = df)
s3 <- summary(cox3)
add_result("3. Crude (Unweighted)",
           s3$coefficients[1, "exp(coef)"],
           s3$conf.int[1, "lower .95"],
           s3$conf.int[1, "upper .95"],
           s3$coefficients[1, "Pr(>|z|)"])

# ============================================================
# 4. Stratified by database
# ============================================================
cat("\n=== 4. STRATIFIED COX (BY DATABASE) ===\n")
cox4 <- coxph(Surv(time_years, event) ~ loneliness + strata(database),
              data = df, weights = iptw, robust = TRUE)
s4 <- summary(cox4)
add_result("4. Stratified (DB)",
           s4$coefficients[1, "exp(coef)"],
           s4$conf.int[1, "lower .95"],
           s4$conf.int[1, "upper .95"],
           s4$coefficients[1, "Pr(>|z|)"])

# ============================================================
# 5. Time-varying effect
# ============================================================
cat("\n=== 5. TIME-VARYING EFFECT ===\n")
cox5 <- coxph(Surv(time_years, event) ~ loneliness + tt(loneliness),
              data = df, weights = iptw, robust = TRUE,
              tt = function(x, t, ...) x * log(t))
s5 <- summary(cox5)
p_tve <- s5$coefficients[2, "Pr(>|z|)"]
cat(paste0("  Time interaction P=", fmt_p(p_tve), "\n"))
add_result("5. Time-varying effect",
           s5$coefficients[1, "exp(coef)"],
           s5$conf.int[1, "lower .95"],
           s5$conf.int[1, "upper .95"],
           s5$coefficients[1, "Pr(>|z|)"])

# ============================================================
# 6. Competing risks (Fine-Gray) — unweighted (cmprsk limitation)
# ============================================================
cat("\n=== 6. COMPETING RISKS (FINE-GRAY) ===\n")
library(cmprsk)
fg <- crr(df$time_years, df$event, df$loneliness, failcode = 1, cencode = 0)
fg_coef <- fg$coef[1]
fg_se <- sqrt(fg$var[1, 1])
fg_hr <- exp(fg_coef)
fg_ci <- exp(fg_coef + c(-1.96, 1.96) * fg_se)
fg_p <- 2 * pnorm(-abs(fg_coef / fg_se))
add_result("6. Fine-Gray (Unweighted)", fg_hr, fg_ci[1], fg_ci[2], fg_p)

# ============================================================
# 7. Multivariable Cox (no IPTW)
# ============================================================
cat("\n=== 7. MULTIVARIABLE COX ===\n")
fmla <- as.formula(paste("Surv(time_years, event) ~ loneliness +",
                         paste(confounders, collapse = " + ")))
cox7 <- coxph(fmla, data = df)
s7 <- summary(cox7)
add_result("7. Multivariable Cox",
           s7$coefficients["loneliness", "exp(coef)"],
           s7$conf.int["loneliness", "lower .95"],
           s7$conf.int["loneliness", "upper .95"],
           s7$coefficients["loneliness", "Pr(>|z|)"])

# ============================================================
# 8. Trimming sensitivity (no trim / 5/95)
# ============================================================
cat("\n=== 8. TRIMMING SENSITIVITY ===\n")
# Reconstruct raw weights from PS
ps_raw <- df$ps_score
ps_raw <- pmin(pmax(ps_raw, 0.001), 0.999)
iptw_raw <- ifelse(df$loneliness == 1, p_treat / ps_raw, (1 - p_treat) / (1 - ps_raw))

# 8a: No trimming
cat("  [No trim]\n")
cox8a <- coxph(Surv(time_years, event) ~ loneliness, data = df,
               weights = iptw_raw, robust = TRUE)
s8a <- summary(cox8a)
add_result("8a. IPTW (No trim)",
           s8a$coefficients[1, "exp(coef)"],
           s8a$conf.int[1, "lower .95"],
           s8a$conf.int[1, "upper .95"],
           s8a$coefficients[1, "Pr(>|z|)"])

# 8b: 5/95 trimming
cat("  [5/95 trim]\n")
lo5 <- quantile(iptw_raw, 0.05)
hi5 <- quantile(iptw_raw, 0.95)
iptw_595 <- pmin(pmax(iptw_raw, lo5), hi5)
cox8b <- coxph(Surv(time_years, event) ~ loneliness, data = df,
               weights = iptw_595, robust = TRUE)
s8b <- summary(cox8b)
add_result("8b. IPTW (5/95 trim)",
           s8b$coefficients[1, "exp(coef)"],
           s8b$conf.int[1, "lower .95"],
           s8b$conf.int[1, "upper .95"],
           s8b$coefficients[1, "Pr(>|z|)"])

# ============================================================
# 9. Leave-one-database-out
# ============================================================
cat("\n=== 9. LEAVE-ONE-DATABASE-OUT ===\n")
for (db_drop in c("HRS", "KLoSA", "SHARE")) {
  sub <- df[df$database != db_drop, ]
  cat(paste0("  [Drop ", db_drop, "]\n"))

  # Fit per-subset PS model
  sub_lr <- glm(as.factor(loneliness) ~ ., data = sub[, c('loneliness', confounders)],
                family = binomial(link = "logit"))
  sub_ps <- predict(sub_lr, type = "response")
  sub_ps <- pmin(pmax(sub_ps, 0.001), 0.999)
  sub_p_treat <- mean(sub$loneliness)
  sub_iptw <- ifelse(sub$loneliness == 1, sub_p_treat / sub_ps,
                     (1 - sub_p_treat) / (1 - sub_ps))
  sub_lo <- quantile(sub_iptw, 0.01)
  sub_hi <- quantile(sub_iptw, 0.99)
  sub_iptw <- pmin(pmax(sub_iptw, sub_lo), sub_hi)

  cox_loo <- coxph(Surv(time_years, event) ~ loneliness, data = sub,
                   weights = sub_iptw, robust = TRUE)
  s_loo <- summary(cox_loo)
  add_result(paste0("9. LOO-", db_drop),
             s_loo$coefficients[1, "exp(coef)"],
             s_loo$conf.int[1, "lower .95"],
             s_loo$conf.int[1, "upper .95"],
             s_loo$coefficients[1, "Pr(>|z|)"])
}

# ============================================================
# 10. PS 1:1 Nearest-neighbor matching
# ============================================================
cat("\n=== 10. PS MATCHING (1:1 NN) ===\n")
set.seed(42)
treat_idx <- which(df$loneliness == 1)
ctrl_idx <- which(df$loneliness == 0)

# Match by PS using nearest neighbor
ps_treat <- df$ps_score[treat_idx]
ps_ctrl <- df$ps_score[ctrl_idx]

matched_ctrl <- integer(length(treat_idx))
for (i in seq_along(treat_idx)) {
  dists <- abs(ps_ctrl - ps_treat[i])
  matched_ctrl[i] <- ctrl_idx[which.min(dists)]
}

matched_idx <- c(treat_idx, matched_ctrl)
df_matched <- df[matched_idx, ]
cat(paste0("  Matched: ", length(treat_idx), " pairs (N=", nrow(df_matched), ")\n"))

cox_match <- coxph(Surv(time_years, event) ~ loneliness, data = df_matched)
s_match <- summary(cox_match)
add_result("10. PS Matching (1:1 NN)",
           s_match$coefficients[1, "exp(coef)"],
           s_match$conf.int[1, "lower .95"],
           s_match$conf.int[1, "upper .95"],
           s_match$coefficients[1, "Pr(>|z|)"])

# ============================================================
# Save all results
# ============================================================
results_df <- do.call(rbind, all_results)
# Format P column
results_df$P_formatted <- sapply(results_df$P, fmt_p)
write.csv(results_df, output_file, row.names = FALSE)
cat(paste0("\nSaved: ", output_file, " (", nrow(results_df), " analyses)\n"))
