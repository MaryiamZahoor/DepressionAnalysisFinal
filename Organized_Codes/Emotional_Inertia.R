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
file_list <- list.files("/media/maryiam/Work/Research/AU to emotion/CREMADExperiments/DAIC_Results/DAIC_Results_NEW/DAIC_RESULTS_RNN_STD/framewise/avg_1_sec", pattern = "*.csv", full.names = TRUE)
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


# ------------------ MLM: Depression LABEL (0/1) ------------------
model_PE <- lmer(PE ~ PE_lag5_c * Depression_Label + (PE_lag5_c | participant_id), data = full_data)
model_NE <- lmer(NE ~ NE_lag5_c * Depression_Label + (NE_lag5_c | participant_id), data = full_data)
model_N  <- lmer(N  ~ N_lag5_c  * Depression_Label + (N_lag5_c  | participant_id), data = full_data)

# One-by-one summaries
summary(model_PE)
summary(model_NE)
summary(model_N)

# Nicely formatted HTML (LABEL)
tab_model(
  model_N,
  dv.labels = "Positive Affect (PE=H)",
  show.re.var = FALSE, show.icc = FALSE, show.se = TRUE, show.r2 = TRUE,
  show.ci = 0.95, digits = 3, string.ci = "95% CI",
  p.val = "satterthwaite", collapse.se = TRUE, string.est = "Estimate (SE)",
  file = "/media/maryiam/Work/Research/R_experiments/Result/MLM/DepressionLabel/1.html"
)

# ------------------ MLM: Depression SEVERITY (centered) ------------------
model_PE_score <- lmer(PE ~ PE_lag5_c * Depression_severity_centered + (PE_lag5_c | participant_id), data = full_data)
model_NE_score <- lmer(NE ~ NE_lag5_c * Depression_severity_centered + (NE_lag5_c | participant_id), data = full_data)
model_N_score  <- lmer(N  ~ N_lag5_c  * Depression_severity_centered + (N_lag5_c  | participant_id), data = full_data)

summary(model_PE_score)
summary(model_NE_score)
summary(model_N_score)

# Nicely formatted HTML (SEVERITY)
tab_model(
  model_N_score,
  dv.labels = "Positive Affect (PE=H)",
  show.re.var = FALSE, show.icc = FALSE, show.se = TRUE, show.r2 = TRUE,
  show.ci = 0.95, digits = 3, string.ci = "95% CI",
  p.val = "satterthwaite", collapse.se = TRUE, string.est = "Estimate (SE)",
  file = "/media/maryiam/Work/Research/R_experiments/Result/MLM/Depression_Severity/PE_NE_N_lag5_severity.html"
)

########################for indivdual emotions########################################
model_H <- lmer(H ~ H_lag5_c * Depression_Label + (H_lag5_c | participant_id), data = full_data)
model_S <- lmer(S ~ S_lag5_c * Depression_Label + (S_lag5_c| participant_id), data = full_data)
model_F <- lmer(F ~ F_lag5_c * Depression_Label + (F_lag5_c | participant_id), data = full_data)
model_D <- lmer(D ~ D_lag5_c * Depression_Label + (D_lag5_c | participant_id), data = full_data)
model_N <- lmer(N ~ N_lag5_c * Depression_Label + (N_lag5_c | participant_id), data = full_data)
model_A <- lmer(A ~ A_lag5_c * Depression_Label + (A_lag5_c | participant_id), data = full_data)

model_H_score <- lmer(H ~ H_lag5_c * Depression_severity_centered + (H_lag5_c | participant_id), data = full_data)
model_S_score <- lmer(S ~ S_lag5_c * Depression_severity_centered  + (S_lag5_c| participant_id), data = full_data)
model_F_score <- lmer(F ~ F_lag5_c * Depression_severity_centered  + (F_lag5_c | participant_id), data = full_data)
model_D_score <- lmer(D ~ D_lag5_c * Depression_severity_centered + (D_lag5_c | participant_id), data = full_data)
model_N_score <- lmer(N ~ N_lag5_c * Depression_severity_centered  + (N_lag5_c | participant_id), data = full_data)
model_A_score <- lmer(A ~ A_lag5_c * Depression_severity_centered  + (A_lag5_c | participant_id), data = full_data) 


summary(model_H)
summary(model_S)
summary(model_F)
summary(model_D)
summary(model_N)
summary(model_A)

summary(model_H_score)
summary(model_S_score)
summary(model_F_score)
summary(model_D_score)
summary(model_N_score)
summary(model_A_score)