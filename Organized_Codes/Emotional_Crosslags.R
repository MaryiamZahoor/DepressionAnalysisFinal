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
# ---------- Packages ----------




###################################################### DONT RUN AGAIN ######################################
# List of participant files
file_list <- list.files("/media/maryiam/Work/Research/AU to emotion/CREMADExperiments/DAIC_Results/DAIC_Results_NEW/DAIC_RESULTS_MLP_UNSTD_COMBINED/avg_1_sec", pattern = "*.csv", full.names = TRUE)
#RNN"/media/maryiam/Work/Research/AU to emotion/CREMADExperiments/DAIC_Results/DAIC_Results_NEW/DAIC_RESULTS_RNN_STD/framewise/avg_1_sec"
#MLP""/media/maryiam/Work/Research/AU to emotion/CREMADExperiments/DAIC_Results/DAIC_Results_NEW/DAIC_RESULTS_MLP/DAIC_RESULTS_MLP_STD/avg_1_sec"

#reading depression labels:
label_file <- fread("/media/maryiam/Work/Research/DAIC Dataset/filtered_participants_corrected_labels.csv")
#label_file <- fread("/media/maryiam/Work/Research/DAIC Dataset/participants_3parts_labeled.csv")
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
  
  
  # Create lag-5 columns
  df[, H_lag5 := shift(H, 5, type = "lag")]
  df[, S_lag5 := shift(S, 5, type = "lag")]
  df[, A_lag5 := shift(A, 5, type = "lag")]
  df[, N_lag5 := shift(N, 5, type = "lag")]
  df[, F_lag5 := shift(F, 5, type = "lag")]
  df[, D_lag5 := shift(D, 5, type = "lag")]
  
  # Group-mean centering of lagged variables by participant
  df[, H_lag5_c := H_lag5 - mean(H, na.rm = TRUE), by = participant_id]
  df[, S_lag5_c := S_lag5 - mean(S, na.rm = TRUE), by = participant_id]
  df[, A_lag5_c := A_lag5 - mean(A, na.rm = TRUE), by = participant_id]
  df[, N_lag5_c := N_lag5 - mean(N, na.rm = TRUE), by = participant_id]
  df[, F_lag5_c := F_lag5 - mean(F, na.rm = TRUE), by = participant_id]
  df[, D_lag5_c := D_lag5 - mean(D, na.rm = TRUE), by = participant_id]
  
  # Remove rows with missing lag values
  df <- na.omit(df)
  
  # Keep only the necessary final columns
  df_small <- df[, .(participant_id, Depression_Label,Depression_severity,Depression_severity_centered,
                     H, H_lag5_c,H_lag5,
                     S, S_lag5_c,S_lag5,
                     A, A_lag5_c,A_lag5,
                     F, F_lag5_c,F_lag5,
                     D, D_lag5_c,D_lag5,
                     N, N_lag5_c,N_lag5)]
  
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


fwrite(full_data, "full_data_unstd.csv")
####################################### RUN FROM HERE #####################################################


full_data <- fread("full_data_unstd.csv")

#########################data descriptive stats#############################################3
{
  # Count the number of samples per participant
  participant_sample_counts <- full_data %>%
    group_by(participant_id) %>%
    summarise(num_samples = n()) %>%
    ungroup()
  
  # View the list of sample counts
  print(participant_sample_counts)
  
  # Calculate the average and standard deviation
  average_samples <- mean(participant_sample_counts$num_samples)
  sd_samples <- sd(participant_sample_counts$num_samples)
  
  # Print the results
  cat("✅ Average number of samples per participant:", average_samples, "\n")
  cat("📊 Standard deviation of samples per participant:", sd_samples, "\n")
}



########################Cross-lags############################

###################compute cross-lag##########################

  
  library(data.table)
  library(lme4)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(stringr)
  #full_data <- fread("full_data.csv")
  # Emotions present in your data
  emotions <- c("H","S","A","F","D","N")
  
  # Predictor names (centered lag-5)
  pred_vars <- paste0(emotions, "_lag5_c")
  
  # Keep only rows with no NAs in the needed columns
  needed <- c("participant_id", emotions, pred_vars)
  dat <- as.data.table(full_data)[, ..needed]
  dat <- na.omit(dat)
  dat[, participant_id := as.factor(participant_id)]
  # ---------- Helper: fit one outcome model and extract per-person slopes ----------
  # Build a model for one outcome; optionally exclude its self-lag (pure cross-lag)
  fit_one_outcome <- function(outcome,
                              cross_only = TRUE,        # TRUE => drop outcome's own lag
                              diagonal_random = TRUE,    # TRUE => '||' random effects (no correlations)
                              maxfun = 2e5) {
    
    preds <- pred_vars
    if (cross_only) {
      preds <- setdiff(preds, paste0(outcome, "_lag5_c"))
    }
    
    # RHS for fixed effects
    rhs <- paste(preds, collapse = " + ")
    
    # Random effects: diagonal (||) or full (|)
    if (diagonal_random) {
      # (0 + p1 || id) + (0 + p2 || id) ... is equivalent to (p1 + p2 || id)
      re_rhs <- paste0("(1 + ", paste(preds, collapse = " || participant_id) + (1 + "), " || participant_id)")
      # Collapse duplicate '|| participant_id) + (0 +' sequence
      re_rhs <- gsub("\\) \\+ \\(1 \\+ \\|\\| participant_id\\)", " || participant_id)", re_rhs, fixed = FALSE)
      # Simpler: one block with all slopes diagonal
      re_rhs <- paste0("(1 + ", paste(preds, collapse = " + "), " || participant_id)")
    } else {
      re_rhs <- paste0("(", rhs, " | participant_id)")
    }
    
    fml <- as.formula(paste0(outcome, " ~ ", rhs, " + ", re_rhs))
    print(fml)
    
    m <- lmer(fml, data = dat, REML = TRUE,
              control = lmerControl(optimizer = "bobyqa",
                                    optCtrl   = list(maxfun = maxfun)))
    
    # Participant-specific coefficients (fixed + random)
    cf <- coef(m)$participant_id                      # matrix/data.frame with columns: (Intercept), preds present
    cf_df <- data.frame(participant_id = rownames(cf), cf, row.names = NULL, check.names = FALSE)
    
    # Keep only slope columns actually present
    slope_cols_present <- intersect(colnames(cf_df), preds)
    
    # Pivot to long
    long <- cf_df |>
      pivot_longer(cols = all_of(slope_cols_present),
                   names_to = "predictor",
                   values_to = "slope") |>
      mutate(outcome      = outcome,
             pred_emotion = sub("_lag5_c$", "", predictor))
    
    # If you want to keep explicit zeros for any dropped predictors (rarely advisable),
    # you could add them here. By default we only keep estimated slopes.
    
    list(model = m, slopes = long)
  }
  # ---------- Fit all 6 outcome models and bind results ----------
  res_list <- map(emotions, ~ fit_one_outcome(.x, cross_only = TRUE, diagonal_random = TRUE))
  all_slopes <- bind_rows(lapply(res_list, `[[`, "slopes"))
  
  # If you want *pure* cross-lag slopes (no inertia), filter them out:
  all_slopes_cross <- all_slopes %>% filter(pred_emotion != outcome)
  
  # Example density (overall) using whatever set you prefer:
  #density_overall <- all_slopes %>%
  #  group_by(participant_id) %>%
  # summarise(density_overall = mean(abs(slope), na.rm = TRUE), .groups = "drop")
  density_overall_cross <- all_slopes_cross %>%
    group_by(participant_id) %>%
    summarise(density_overall_cross = mean(abs(slope), na.rm = TRUE), .groups = "drop")
  
  # Positive / Negative sets by outcome (adjust to your coding if needed)
  pos_outcomes <- c("H")           # positive emotions (as outcome)
  neg_outcomes <- c("A","F","D","S")  # negative emotions (as outcome)
  # Neutral excluded from +/- densities:
  # pos/neg density uses all predictors, but only outcomes in the set.
  
  density_pos <- all_slopes_cross |>
    filter(outcome %in% pos_outcomes) |>
    group_by(participant_id) |>
    summarise(density_pos = mean(abs(slope), na.rm = TRUE), .groups = "drop")
  
  density_neg <- all_slopes_cross |>
    filter(outcome %in% neg_outcomes) |>
    group_by(participant_id) |>
    summarise(density_neg = mean(abs(slope), na.rm = TRUE), .groups = "drop")
  
  # ---------- Combine densities ----------
  densities <- density_overall_cross |>
    left_join(density_pos, by = "participant_id") |>
    left_join(density_neg, by = "participant_id")
  
  # Optional: join back labels for group comparisons
  if (all(c("Depression_Label","Depression_severity") %in% names(full_data))) {
    library(dplyr)
    labz <- unique(full_data[, .(participant_id, Depression_Label,Depression_severity, Depression_severity_centered)])
    labz[, participant_id := as.character(participant_id)]  # data.table mutate
    densities <- densities %>%
      mutate(participant_id = as.character(participant_id)) %>%
      left_join(labz, by = "participant_id")
  }
  densities <- densities %>%
    mutate(Depression_Label = factor(Depression_Label, levels = c("0","1")))
  ###############indivdual emotion densities##########33333
 { # all_slopes_cross already exists (outcome, pred_emotion, slope, participant_id)
  per_emotion_density <- all_slopes_cross %>%
    group_by(participant_id, outcome) %>%
    summarise(
      density_emotion = mean(abs(slope), na.rm = TRUE),
      .groups = "drop"
    )
  
  # Add labels and severity from full_data
  labz <- unique(full_data[, .(participant_id, Depression_Label, Depression_severity_centered)])
  labz[, participant_id := as.character(participant_id)]
  
  per_emotion_density <- per_emotion_density %>%
    mutate(participant_id = as.character(participant_id)) %>%
    left_join(labz, by = "participant_id")
  emotions <- unique(per_emotion_density$outcome)
  
    }

  ####################Depression Label########################
  {
    #############################Compute stats per Dep Label##############
    library(dplyr)
    library(tidyr)
    
    # --- edit these to match your objects/columns ---
    # densities: per-participant overall/pos/neg
    vars_overall <- c("density_overall_cross", "density_pos", "density_neg")
    
    # ---- 1) summaries for overall / pos / neg ----
    overall_tbl <- densities %>%
      pivot_longer(all_of(vars_overall), names_to = "metric", values_to = "value") %>%
      group_by(Depression_Label, metric) %>%
      summarise(
        n      = sum(is.finite(value)),
        mean   = mean(value, na.rm = TRUE),
        sd     = sd(value, na.rm = TRUE),
        se     = sd / sqrt(n),
        median = median(value, na.rm = TRUE),
        IQR    = IQR(value, na.rm = TRUE),
        .groups = "drop"
      )
    
    # ---- 2) summaries for per-emotion matrix (emo_dens) ----
    # expecting columns: participant_id, outcome (emotion code), density_emotion, Depression_Label, ...
    emo_tbl <- per_emotion_density %>%
      rename(metric = outcome, value = density_emotion) %>%
      mutate(Depression_Label = factor(Depression_Label, levels = c("0","1"))) %>%
      group_by(Depression_Label, metric) %>%
      summarise(
        n      = sum(is.finite(value)),
        mean   = mean(value, na.rm = TRUE),
        sd     = sd(value, na.rm = TRUE),
        se     = sd / sqrt(n),
        median = median(value, na.rm = TRUE),
        IQR    = IQR(value, na.rm = TRUE),
        .groups = "drop"
      )
    
    # ---- 3) unified table ----
    summary_all <- bind_rows(overall_tbl, emo_tbl) %>%
      arrange(metric, Depression_Label)
    
    summary_all
  }
  ##############Dperession Label#########################
  #library(robustbase)
  library(lmtest)
  library(MASS)
  library(sandwich)
  densities$Depression_Label <- as.integer(densities$Depression_Label) - 1
  #y <- as.integer(densities$Depression_Label) - 1
  
  densities$density_z <- as.numeric(scale(densities$density_overall_cross))
  densities$density_z_neg <- as.numeric(scale(densities$density_neg))
  densities$density_z_pos <- as.numeric(scale(densities$density_pos))
  
  
  robust_mod=rlm(density_overall_cross ~ Depression_Label, data = densities)
  summary(robust_mod)
  robust_vcov <- vcovHC(robust_mod, type = "HC0")
  coeftest(robust_mod, vcov = robust_vcov)
  confint.default(robust_mod , vcov. = rob_vcov) 
  
  robust_mod=rlm(density_neg ~ Depression_Label, data = densities)
  summary(robust_mod)
  robust_vcov <- vcovHC(robust_mod, type = "HC0")
  coeftest(robust_mod, vcov = robust_vcov)
  confint.default(robust_mod , vcov. = rob_vcov)
  
  robust_mod=rlm(density_pos ~ Depression_Label, data = densities)
  summary(robust_mod)
  robust_vcov <- vcovHC(robust_mod, type = "HC0")
  coeftest(robust_mod, vcov = robust_vcov)
  confint.default(robust_mod , vcov. = rob_vcov)
  
  robust_mod=rlm(density_overall_cross ~ Depression_severity_centered, data = densities)
  summary(robust_mod)
  robust_vcov <- vcovHC(robust_mod, type = "HC0")
  coeftest(robust_mod, vcov = robust_vcov)
  confint.default(robust_mod , vcov. = rob_vcov) 
  
  robust_mod=rlm(density_neg ~ Depression_severity_centered, data = densities)
  summary(robust_mod)
  robust_vcov <- vcovHC(robust_mod, type = "HC0")
  coeftest(robust_mod, vcov = robust_vcov)
  confint.default(robust_mod , vcov. = rob_vcov)
  
  robust_mod=rlm(density_pos ~ Depression_severity_centered, data = densities)
  summary(robust_mod)
  robust_vcov <- vcovHC(robust_mod, type = "HC0")
  coeftest(robust_mod, vcov = robust_vcov)
  confint.default(robust_mod , vcov. = rob_vcov)
  
  
  #############indivdual emotion densities#############################################
  emo_df <- per_emotion_density %>%
    transmute(
      outcome = factor(outcome),                  # e.g., H,S,A,D,F,N
      y = as.numeric(density_emotion),
      sev = as.numeric(Depression_severity_centered),
      label=as.numeric(Depression_Label)
    ) %>%
    filter(is.finite(y), is.finite(sev))
  
  # (Optional) z-score severity within the whole sample for per-SD interpretation
  # emo_df <- emo_df %>% mutate(sev_z = as.numeric(scale(sev)))
  
  # --- 1) Fit rlm per emotion:  density_emotion ~ Depression_severity ---
  fit_one <- function(df) {
    fit <- rlm(y ~ sev, data = df)  
    fit2 <- rlm(y ~ label, data = df) 
    
    #print(ydf)# <- THIS is the model you asked for
    rob_vcov <- vcovHC(fit, type = "HC0")        # robust (HC) SEs
    ct <- coeftest(fit, vcov = rob_vcov)
    
    rob_vcov1 <- vcovHC(fit2, type = "HC0")        # robust (HC) SEs
    ct1 <- coeftest(fit2, vcov = rob_vcov1)
    
    
    # Extract slope (severity effect)
    b  <- unname(ct["sev","Estimate"])
    se <- unname(ct["sev","Std. Error"])
    z  <- unname(ct["sev","z value"])
    p  <- unname(ct["sev","Pr(>|z|)"])
    
    # Extract slope (severity effect)
    b1  <- unname(ct1["label","Estimate"])
    se1 <- unname(ct1["label","Std. Error"])
    z1  <- unname(ct1["label","z value"])
    p1  <- unname(ct1["label","Pr(>|z|)"])
    
    
    # 95% CI from robust SEs
    ci_lo <- b - 1.96*se
    ci_hi <- b + 1.96*se
    
    # 95% CI from robust SEs
    ci_lo1 <- b1 - 1.96*se1
    ci_hi1 <- b1 + 1.96*se1
    
    tibble::tibble(
      n = nrow(df),
      B = b, SE = se, z = z, p = p,
      CI_low = ci_lo, CI_high = ci_hi,
      n_label = nrow(df),
      B_label = b1, SE_label = se1, z_label = z1, p_label = p1,
      CI_low_label = ci_lo1, CI_high_label  = ci_hi1,
      # resid_SD = fit2$s                           # robust residual SD
    )
  }
  
  # --- 2) Run for each emotion & combine ---
  rlm_by_emotion <- emo_df %>%
    group_split(outcome) %>%
    map_dfr(~ fit_one(.x) %>% mutate(outcome = unique(.x$outcome))) %>%
    relocate(outcome)
  
  rlm_by_emotion
  
  
