library(effectsize)
library(ggplot2)
library(car)
library(data.table)
library(dplyr)
library(lme4)
library(lmerTest)
library(psych)
library(data.table)
library(coin)
library(nnet)
library(pivottabler)
library(apaTables)
library(sjPlot) 



###################################################### DONT RUN AGAIN ######################################
# List of participant files
file_list <- list.files("/media/[REDACTED]/Work/Research/AU to emotion/CREMADExperiments/DAIC_Results/DAIC_Results_NEW/DAIC_RESULTS_RNN_STD/framewise/avg_1_sec", pattern = "*.csv", full.names = TRUE)
#RNN"/media/[REDACTED]/Work/Research/AU to emotion/CREMADExperiments/DAIC_Results/DAIC_Results_NEW/DAIC_RESULTS_RNN_STD/framewise/avg_1_sec"
#MLP""/media/[REDACTED]/Work/Research/AU to emotion/CREMADExperiments/DAIC_Results/DAIC_Results_NEW/DAIC_RESULTS_MLP/DAIC_RESULTS_MLP_STD/avg_1_sec"

#reading depression labels:
label_file <- fread("/media/[REDACTED]/Work/Research/DAIC Dataset/filtered_participants_corrected_labels.csv")
#label_file <- fread("/media/[REDACTED]/Work/Research/DAIC Dataset/participants_3parts_labeled.csv")
#participants_3parts_labeled, participants_corrected_labeled , equal_participant_randomized_5
#participants_5_10_labeled , participants_all_thresholds_labeled
label_file[, Depression_severity_centered := Depression_severity - mean(Depression_severity, na.rm = TRUE)]

# Ensure proper column names

setnames(label_file, old = "Participant", new = "Subject_ID")

label_ids <- as.character(unique(label_file$Subject_ID))

# Helper: first 3 chars of the basename are the Subject_ID
extract_id <- function(path) substr(basename(path), 1, 3)

skipped   <- 0L
processed <- 0L
all_data  <- list()
# Initialize an empty list to collect data
all_data <- list()

# Loop over files efficiently
for (file in file_list) {
  # Read only the needed columns using fread and select
  sid <- extract_id(file)
  
  # Skip early if this Subject_ID is not in the label file
  if (is.na(sid) || !(sid %in% label_ids)) {
    message("Skipping (no matching label): ", basename(file), " [Subject_ID=", sid, "]")
    skipped <- skipped + 1L
    next
  }
  df <- fread(file, select = c("Subject_ID", "Depression_Label", "A","N","F","H", "D","S"))
  # Add correct depression label by joining with label_file
  
  df <- merge(df, label_file[, .(Subject_ID, corrected_label = Depression_label,Depression_severity, Depression_severity_centered)],
              by = "Subject_ID", all.x = TRUE)
  
  
  # Replace original Depression_Label with corrected one
  df[, Depression_Label := corrected_label]
  df[, corrected_label := NULL]  # remove the temporary column
  
  # Skip if required columns are missing
  #if (!all(c("Subject_ID", "Depression_Label", "PE", "NE") %in% names(df))) next
  
  # Add participant_id column
  df$participant_id <- df$Subject_ID
  # -------------------- NEW: PE / NE definitions --------------------
  # PE: happiness only
  df[, PE := H]
  
  # NE: mean of negative-valence emotions (S, A, F, D)
  # (If you prefer SUM, replace the next line with: NE := S + A + F + D)
  df[, NE := rowMeans(.SD, na.rm = TRUE), .SDcols = c("S","A","F","D")]
  
  k <- 5
  # original emotions
  df[, H_lag5 := shift(H, k, type = "lag")]
  df[, S_lag5 := shift(S, k, type = "lag")]
  df[, A_lag5 := shift(A, k, type = "lag")]
  df[, N_lag5 := shift(N, k, type = "lag")]
  df[, F_lag5 := shift(F, k, type = "lag")]
  df[, D_lag5 := shift(D, k, type = "lag")]
  
  # NEW: PE/NE lags
  df[, PE_lag5 := shift(PE, k, type = "lag")]
  df[, NE_lag5 := shift(NE, k, type = "lag")]
  
  
  # Group-mean centering of lagged variables by participant
  df[, H_lag5_c := H_lag5 - mean(H, na.rm = TRUE), by = participant_id]
  df[, S_lag5_c := S_lag5 - mean(S, na.rm = TRUE), by = participant_id]
  df[, A_lag5_c := A_lag5 - mean(A, na.rm = TRUE), by = participant_id]
  df[, N_lag5_c := N_lag5 - mean(N, na.rm = TRUE), by = participant_id]
  df[, F_lag5_c := F_lag5 - mean(F, na.rm = TRUE), by = participant_id]
  df[, D_lag5_c := D_lag5 - mean(D, na.rm = TRUE), by = participant_id]
  
  # NEW: center PE/NE lags by participant means of PE/NE
  df[, PE_lag5_c := PE_lag5 - mean(PE_lag5, na.rm = TRUE), by = participant_id]
  df[, NE_lag5_c := NE_lag5 - mean(NE_lag5, na.rm = TRUE), by = participant_id]
  # Remove rows with missing lag values
  df <- na.omit(df)
  
  # Keep only the necessary final columns
  df_small <- df[, .(
    participant_id, Depression_Label, Depression_severity, Depression_severity_centered,
    # individual emotions
    H, H_lag5, H_lag5_c,
    S, S_lag5, S_lag5_c,
    A, A_lag5, A_lag5_c,
    F, F_lag5, F_lag5_c,
    D, D_lag5, D_lag5_c,
    N, N_lag5, N_lag5_c,
    # NEW aggregates
    PE, PE_lag5, PE_lag5_c,
    NE, NE_lag5, NE_lag5_c
  )]
  
  # Add to list
  all_data[[file]] <- df_small
}

rm(df, df_small)
gc()
# Combine all into one data.frame
full_data <- rbindlist(all_data)
full_data <- full_data %>%
  group_by(participant_id) %>%
  mutate(timepoint = row_number()) %>%
  ungroup()


fwrite(full_data, "full_data.csv")
####################################### RUN FROM HERE #####################################################


full_data <- fread("full_data.csv")
variability_df <- full_data %>%
  group_by(participant_id, Depression_Label,Depression_severity) %>%
  summarise(
    H_variability = sd(H, na.rm = TRUE),
    S_variability = sd(S, na.rm = TRUE),
    D_variability = sd(D, na.rm = TRUE),
    F_variability = sd(F, na.rm = TRUE),
    N_variability = sd(N, na.rm = TRUE),
    A_variability = sd(A, na.rm = TRUE),
    PE_variability = sd(PE, na.rm = TRUE),
    NE_variability = sd(NE, na.rm = TRUE),
    
  )
# Make sure the grouping variable is a factor with your intended reference level
# (e.g., 0 = healthy, 1 = depressed)
variability_df <- variability_df %>%
  mutate(Depression_Label = factor(Depression_Label, levels = c(0, 1)))

vars <- c("H_variability","S_variability","D_variability","F_variability","N_variability","A_variability",
          "PE_variability","NE_variability")

res_table <- map_dfr(vars, function(v) {
  fmla <- reformulate("Depression_Label", response = v)
  print(fmla)
  
  # Wilcoxon (normal approximation; no continuity correction — consistent for large n / ties)
  w <- wilcox.test(fmla, data = variability_df, exact = FALSE, correct = FALSE,conf.int = TRUE, conf.level = 0.95)
  
  # Rank-biserial correlation (point estimate + 95% CI)
  rb <- rank_biserial(fmla, data = variability_df, ci = 0.95)
  #print(rb)
  #print(rb$r_rank_biserial)
  
  # Optional: Cliff's delta + A12 (common-language)
  #cd <- tryCatch(cliff_delta(fmla, data = variability_df, ci = 0.95), error = function(e) NULL)
  
  tibble(
    outcome   = v,
    #n_group0  = sum(variability_df$Depression_Label == levels(variability_df$Depression_Label)[1] &
    #                 !is.na(variability_df[[v]])),
    #n_group1  = sum(variability_df$Depression_Label == levels(variability_df$Depression_Label)[2] &
    #                 !is.na(variability_df[[v]])),
    W         = unname(w$statistic),
    ci_low    = unname(as.numeric(w$conf.int[1])),   # 95% CI lower bound
    ci_high   = unname(as.numeric(w$conf.int[2])),
    p         = w$p.value,
    rb      = rb$r_rank_biserial,
    #r_rb_low  = rb$CI_low,
    #r_rb_high = rb$CI_high,
    
    # cliff     = if (!is.null(cd)) cd$Delta else NA_real_,
    #  cliff_low = if (!is.null(cd)) cd$CI_low else NA_real_,
    #  cliff_high= if (!is.null(cd)) cd$CI_high else NA_real_,
    #  A12       = if (!is.null(cd)) (cd$Delta + 1) / 2 else NA_real_
  )
  
})

res_table
res_table <- map_dfr(vars, function(v) {
  # build formula: outcome ~ Depression_Label
  fmla <- as.formula(paste(v, "~ Depression_Label"))
  print(fmla)
  
  # robust regression with Huber psi
  robust_mod  <- rlm(fmla, data = variability_df, psi = psi.huber)
  # robust sandwich covariance
  robust_vcov <- vcovHC(robust_mod, type = "HC0")
  
  # robust coef test table
  ct <- coeftest(robust_mod, vcov = robust_vcov)
  
  # small helper to safely extract values
  take <- function(mat, row, col) unname(mat[row, col, drop = TRUE])
  
  # slope for Depression_Label
  est <- take(ct, "Depression_Label1", "Estimate")
  se  <- take(ct, "Depression_Label1", "Std. Error")
  z   <- take(ct, "Depression_Label1", "z value")
  p   <- take(ct, "Depression_Label1", "Pr(>|z|)")
  
  # 95% CI (same as confint.default(…, vcov.=robust_vcov))
  ci_low  <- est - 1.96 * se
  ci_high <- est + 1.96 * se
  
  tibble(
    outcome = v,
    term    = "Depression_Label",
    n       = stats::nobs(robust_mod),
    estimate = est,
    se       = se,
    z        = z,
    p        = p,
    ci_low   = ci_low,
    ci_high  = ci_high
  )
})

res_table

###################################depression severity###################################3
library(MASS)
library(sandwich)
library(lmtest)
library(dplyr)
library(purrr)
library(tibble)

# Outcomes to run (add PE_intensity / NE_intensity if you computed them)
outs <- c("H_variability","S_variability","D_variability","F_variability","N_variability","A_variability",
          "PE_variability","NE_variability")

fit_rlm_huber <- function(y, df = variability_df) {
  fml <- as.formula(paste(y, "~ Depression_severity"))
  mr  <- rlm(fml, data = df, psi = psi.huber)
  ct  <- coeftest(mr, vcov = vcovHC(mr, type = "HC0"))
  

  # extract rows safely
  take <- function(mat, row, col) unname(mat[row, col, drop = TRUE])
  
  # Intercept
  #b0  <- take(ct, "(Intercept)", "Estimate")
  #se0 <- take(ct, "(Intercept)", "Std. Error")
  #z0  <- take(ct, "(Intercept)", "z value")
  #p0  <- take(ct, "(Intercept)", "Pr(>|z|)")
  
  # Slope (Depression_severity)
  b1  <- take(ct, "Depression_severity", "Estimate")
  se1 <- take(ct, "Depression_severity", "Std. Error")
  z1  <- take(ct, "Depression_severity", "z value")
  p1  <- take(ct, "Depression_severity", "Pr(>|z|)")
  
  tibble(
    outcome = y,
    n = stats::nobs(mr),
    term ="Depression_severity",
    estimate = b1,
    se = se1,
    z = z1,
    p = p1,
    ci_low = estimate - 1.96 * se,
    ci_high = estimate + 1.96 * se
  )
}

rlm_results <- map_dfr(outs, fit_rlm_huber)

# View all results
rlm_results

library(MASS)
library(lmtest)
library(sandwich)
library(dplyr)
library(purrr)
library(tibble)

fit_rlm_huber <- function(y, df = variability_df) {
  fml <- as.formula(paste(y, "~ Depression_severity"))
  
  # --- Robust linear model (Huber M-estimation) ---
  mr  <- rlm(fml, data = df, psi=psi.bisquare)
  ct_r <- coeftest(mr, vcov = vcovHC(mr, type = "HC0"))
  
  take <- function(mat, row, col) unname(mat[row, col, drop = TRUE])
  
  # RLM slope
  b1_r  <- take(ct_r, "Depression_severity", "Estimate")
  se1_r <- take(ct_r, "Depression_severity", "Std. Error")
  z1_r  <- take(ct_r, "Depression_severity", "z value")
  p1_r  <- take(ct_r, "Depression_severity", "Pr(>|z|)")
  
  # --- Ordinary Linear Model (OLS) ---
  ml  <- lm(fml, data = df)
  ct_l <- coeftest(ml, vcov = vcovHC(ml, type = "HC0"))
  
  # OLS slope
  b1_l  <- take(ct_l, "Depression_severity", "Estimate")
  se1_l <- take(ct_l, "Depression_severity", "Std. Error")
  z1_l  <- take(ct_l, "Depression_severity", "t value") # note: t ~ z
  p1_l  <- take(ct_l, "Depression_severity", "Pr(>|t|)")
  
  # --- Combine both results ---
  tibble(
    outcome = y,
    n = nobs(mr),
    term = "Depression_severity",
    
    # Robust model results
    estimate_rlm = b1_r,
    se_rlm = se1_r,
    z_rlm = z1_r,
    p_rlm = p1_r,
    ci_low_rlm = b1_r - 1.96 * se1_r,
    ci_high_rlm = b1_r + 1.96 * se1_r,
    
    # OLS model results
    estimate_lm = b1_l,
    se_lm = se1_l,
    z_lm = z1_l,
    p_lm = p1_l,
    ci_low_lm = b1_l - 1.96 * se1_l,
    ci_high_lm = b1_l + 1.96 * se1_l
  )
}

# Run for all outcomes
rlm_results <- map_dfr(outs, fit_rlm_huber)

rlm_results
