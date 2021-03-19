# Script for Initial Amyloid Model Fits and Cross-Validation

library(readr)
library(dplyr)
library(stringr)
library(origami)

# Read in Data & Modules ####
beta_tau <- read_csv("AmyloidData/beta_tau_csf_new_vars.csv")
source("cv_functions.R")

# Filter By Visit ####
baseline <- beta_tau %>%
  filter(VISCODE == "bl")
month12 <- beta_tau %>%
  filter(VISCODE == "m12")
month24 <- beta_tau %>%
  filter(VISCODE == "m24")

# Filter out NAs for Separate Models ####
bl_na1 <- which(is.na(baseline$beta_pos) | is.na(baseline$RAVLT_immediate_bl) | 
                  is.na(baseline$Hippocampus) | is.na(baseline$WholeBrain))
m12_na1 <- which(is.na(month12$beta_pos) | is.na(month12$RAVLT_immediate) | 
                   is.na(month12$Hippocampus) | is.na(month12$WholeBrain))
m24_na1 <- which(is.na(month24$beta_pos) | is.na(month24$RAVLT_immediate) | 
                   is.na(month24$Hippocampus) | is.na(month24$WholeBrain))

bl_na2 <- which(is.na(baseline$beta_pos) | is.na(baseline$RAVLT_immediate_bl))
m12_na2 <- which(is.na(month12$beta_pos) | is.na(month12$RAVLT_immediate))
m24_na2 <- which(is.na(month24$beta_pos) | is.na(month24$RAVLT_immediate))

# Rename _bl Variables ####
bl_rm <- c(27, 54, 55)
baseline_1 <- baseline[-(bl_na1), -(bl_rm)]
baseline_1 <- rename(
  baseline_1, 
  RAVLT_immediate = RAVLT_immediate_bl,
  Hippocampus = Hippocampus_bl,
  WholeBrain = WholeBrain_bl)

month12_1 <- month12[-(m12_na1), ]
month24_1 <- month24[-(m24_na1), ]

baseline_2 <- baseline[-(bl_na2), -(bl_rm)]
baseline_2 <- rename(
  baseline_2, 
  RAVLT_immediate = RAVLT_immediate_bl,
  Hippocampus = Hippocampus_bl,
  WholeBrain = WholeBrain_bl)

month12_2 <- month12[-(m12_na2), ]
month24_2 <- month24[-(m24_na2), ]

# Initial Model Fits ####
# Including MRI data
bl_cont <- lm(beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain, 
              data = baseline_1)
bl_logit <- glm(beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain, 
                data = baseline_1, family = "binomial")
m12_cont <- lm(beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain, 
               data = month12_1)
m12_logit <- glm(beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain, 
                 data = month12_1, family = "binomial")
m24_cont <- lm(beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain, 
               data = month24_1)
m24_logit <- glm(beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain, 
                 data = month24_1, family = "binomial")

# Excluding MRI Data
bl_cont2 <- lm(beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate, 
               data = baseline_2)
bl_logit2 <- glm(beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate, 
                 data = baseline_2, family = "binomial")
m12_cont2 <- lm(beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate, 
                data = month12_1)
m12_logit2 <- glm(beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate, 
                  data = month12_1, family = "binomial")
m24_cont2 <- lm(beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate, 
                data = month24_1)
m24_logit2 <- glm(beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate, 
                  data = month24_1, family = "binomial")


# 5-Fold Cross Validation ####
# Make Folds
bl_folds1 <- folds_vfold(nrow(baseline_1), V = 5)
m12_folds1 <- folds_vfold(nrow(month12_1), V = 5)
m24_folds1 <- folds_vfold(nrow(month24_1), V = 5)

bl_folds2 <- folds_vfold(nrow(baseline_2), V = 5)
m12_folds2 <- folds_vfold(nrow(month12_2), V = 5)
m24_folds2 <- folds_vfold(nrow(month24_2), V = 5)

# Linear Models (Baseline, 12 Month, 24 Month)
bl_cv_lm <- origami::cross_validate(
  cv_fun = cv_lm, folds = bl_folds1, data = baseline_1,
  reg_form = "beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain")

m12_cv_lm <- origami::cross_validate(
  cv_fun = cv_lm, folds = m12_folds1, data = month12_1,
  reg_form = "beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain")

m24_cv_lm <- origami::cross_validate(
  cv_fun = cv_lm, folds = m24_folds1, data = month24_1,
  reg_form = "beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain")

bl_cv_lm2 <- origami::cross_validate(
  cv_fun = cv_lm, folds = bl_folds2, data = baseline_2,
  reg_form = "beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate")

m12_cv_lm2 <- origami::cross_validate(
  cv_fun = cv_lm, folds = m12_folds2, data = month12_2,
  reg_form = "beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate")

m24_cv_lm2 <- origami::cross_validate(
  cv_fun = cv_lm, folds = m24_folds2, data = month24_2,
  reg_form = "beta_csf ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate")

# Logistic Models (Baseline, 12 Month, 24 Month)
bl_cv_logit <- origami::cross_validate(
  cv_fun = cv_logit, folds = bl_folds1, data = baseline_1,
  reg_form = "beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain")

m12_cv_logit <- origami::cross_validate(
  cv_fun = cv_logit, folds = m12_folds1, data = month12_1,
  reg_form = "beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain")

m24_cv_logit <- origami::cross_validate(
  cv_fun = cv_logit, folds = m24_folds1, data = month24_1,
  reg_form = "beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate + Hippocampus + WholeBrain")

bl_cv_logit2 <- origami::cross_validate(
  cv_fun = cv_logit, folds = bl_folds2, data = baseline_2,
  reg_form = "beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate")

m12_cv_logit2 <- origami::cross_validate(
  cv_fun = cv_logit, folds = m12_folds2, data = month12_2,
  reg_form = "beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate")

m24_cv_logit2 <- origami::cross_validate(
  cv_fun = cv_logit, folds = m24_folds2, data = month24_2,
  reg_form = "beta_pos ~ AGE + MALE + APOE4_1 + APOE4_2 + RAVLT_immediate")

# Linear Model Stats
bl_lm_stats1 <- colMeans(dplyr::bind_rows(bl_cv_lm$c_stats))
m12_lm_stats1 <- colMeans(dplyr::bind_rows(m12_cv_lm$c_stats))
m24_lm_stats1 <- colMeans(dplyr::bind_rows(m24_cv_lm$c_stats))

bl_lm_stats2 <- colMeans(dplyr::bind_rows(bl_cv_lm2$c_stats))
m12_lm_stats2 <- colMeans(dplyr::bind_rows(m12_cv_lm2$c_stats))
m24_lm_stats2 <- colMeans(dplyr::bind_rows(m24_cv_lm2$c_stats))

# Logistic Model Stats
bl_log_stats1 <- colMeans(dplyr::bind_rows(bl_cv_logit$c_stats))
m12_log_stats1 <- colMeans(dplyr::bind_rows(m12_cv_logit$c_stats))
m24_log_stats1 <- colMeans(dplyr::bind_rows(m24_cv_logit$c_stats))

bl_log_stats2 <- colMeans(dplyr::bind_rows(bl_cv_logit2$c_stats))
m12_log_stats2 <- colMeans(dplyr::bind_rows(m12_cv_logit2$c_stats))
m24_log_stats2 <- colMeans(dplyr::bind_rows(m24_cv_logit2$c_stats))

# Combine Data
bl_summary <- bind_rows(bl_lm_stats1, bl_lm_stats2, bl_log_stats1, bl_log_stats2)
m12_summary <- bind_rows(m12_lm_stats1, m12_lm_stats2, m12_log_stats1, m12_log_stats2)
m24_summary <- bind_rows(m24_lm_stats1, m24_lm_stats2, m24_log_stats1, m24_log_stats2)

