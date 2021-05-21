# Updated Amyloid Positivity Models

library(readr)
library(dplyr)
library(stringr)
library(origami)
library(glmnet)
library(ROCR)
library(ggpubr)
library(stargazer)
source("R/cv_functions.R")

# Read in Amyloid Positivity Data
amyloid_pos <- read_csv("processed_data/amyloid_pos_data.csv",
                        col_types = cols(
                          AV45_bl = col_double(),
                          av45_SUVR_bl = col_double(),
                          br_vol_bl = col_double()))

                  

# Filter by Baseline, Non-NA, and Holdout 
baseline <- amyloid_pos %>% 
  dplyr::filter(VISCODE == "bl", 
                is.na(DX) == F, 
                is.na(AGE) == F)

# Subset Holdout Group
holdout <- baseline %>% filter(Holdout == 1)
baseline <- baseline %>% filter(Holdout == 0, is.na(RAVLT_immediate_bl) == F)


# Filter Out AD Patients to Predict Amyloid Positivity - Diimensions 1,178 x 57 OLD###
# Filter Out AD Patients to Predict Amyloid Positivity - Diimensions 1295 x 57 NEW###
noAD <- baseline %>% filter(DX_bl != "AD")
# Subset by Patients With Genetic Data Available - Dimensions 1,148 x 57 OLD ###
# Subset by Patients With Genetic Data Available - Dimensions 1265 x 57 NEW ###
hasAP <- noAD[!is.na(noAD$APOE4_1), ]



# Make Folds - 5 Fold CV
set.seed(12345)
folds <- folds_vfold(nrow(noAD), V = 5)
foldsAP <- folds_vfold(nrow(hasAP), V = 5)

# Logistic Models for Individual Questions - 0.5 Cutoff
mmse_logit <- origami::cross_validate(
  cv_fun = cv_logit, folds = folds, data = noAD, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ MMSE_bl")

ldel_logit <- origami::cross_validate(
  cv_fun = cv_logit, folds = folds, data = noAD, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ LDELTOTAL_BL")

adas_logit <- origami::cross_validate(
  cv_fun = cv_logit, folds = folds, data = noAD, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ ADASQ4_bl")

ravlt_logit <- origami::cross_validate(
  cv_fun = cv_logit, folds = folds, data = noAD, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ RAVLT_immediate_bl")



# Logistic Models 0.5 Cutoff
cv5_logit1 <- origami::cross_validate(
  cv_fun = cv_logit, folds = folds, data = noAD, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ AGE + Male + MMSE_bl")

cv5_logit2 <- origami::cross_validate(
  cv_fun = cv_logit, folds = folds, data = noAD, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL")

cv5_logit3 <- origami::cross_validate(
  cv_fun = cv_logit, folds = folds, data = noAD, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl")

cv5_logit4 <- origami::cross_validate(
  cv_fun = cv_logit, folds = folds, data = noAD, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl + RAVLT_immediate_bl")

cv5_logit5 <- origami::cross_validate(
  cv_fun = cv_logit, folds = foldsAP, data = hasAP, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2")

cv5_logit6 <- origami::cross_validate(
  cv_fun = cv_logit, folds = foldsAP, data = hasAP, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2 + PTEDUCAT")

cv5_logit7 <- origami::cross_validate(
  cv_fun = cv_logit, folds = foldsAP, data = hasAP, cutoff = 0.5,
  reg_form = "beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2 + AGE*RAVLT_immediate_bl*CollegePlus + AGE*MMSE_bl*CollegePlus")


mmse <- colMeans(dplyr::bind_rows(mmse_logit$c_stats))
ldel <- colMeans(dplyr::bind_rows(ldel_logit$c_stats))
adas <- colMeans(dplyr::bind_rows(adas_logit$c_stats))
ravlt <- colMeans(dplyr::bind_rows(ravlt_logit$c_stats))

quests <- dplyr::bind_rows(mmse, ldel, adas, ravlt)

cv1 <- colMeans(dplyr::bind_rows(cv5_logit1$c_stats))
cv2 <- colMeans(dplyr::bind_rows(cv5_logit2$c_stats))
cv3 <- colMeans(dplyr::bind_rows(cv5_logit3$c_stats))
cv4 <- colMeans(dplyr::bind_rows(cv5_logit4$c_stats))
cv5 <- colMeans(dplyr::bind_rows(cv5_logit5$c_stats))
cv6 <- colMeans(dplyr::bind_rows(cv5_logit6$c_stats))
cv7 <- colMeans(dplyr::bind_rows(cv5_logit7$c_stats))

cvs <- dplyr::bind_rows(cv1, cv2, cv3, cv5, cv5, cv6, cv7)


mmse.fit <- glm(beta_pos_vote ~ MMSE_bl, data = hasAP, family = "binomial")
ldel.fit <- glm(beta_pos_vote ~ LDELTOTAL_BL, data = hasAP, family = "binomial")
adas.fit <- glm(beta_pos_vote ~ ADASQ4_bl, data = hasAP, family = "binomial")
ravlt.fit <- glm(beta_pos_vote ~ RAVLT_immediate_bl, data = hasAP, family = "binomial")

question_table <- stargazer(mmse.fit, ldel.fit, adas.fit, ravlt.fit,
                            header = FALSE,
                            title = "Cognitive Assessments",
                            covariate.labels = c("MMSE", "LDEL", "ADAS", "RAVLT"),
                            dep.var.labels = c("beta-amyloid Positive"))
readr::write_lines(question_table, "figures/quest_tab.tex")

full_tab <- stargazer(fit1, fit2, fit3, fit4, fit5, fit6,
                      header = FALSE,
                      title = "Logistic Models",
                      covariate.labels = c("Age", "Male", 
                                           "MMSE", "LDEL", "ADAS", "RAVLT",
                                           "APOE4 - 1", "APOE4 - 2", 
                                           "Educ"),
                      dep.var.labels = c("beta-amyloid Positive"))
readr::write_lines(full_tab, "figures/stage1.tex")

# Full Fits
fit1 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl, data = noAD, family = "binomial")
fit2 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL, data = noAD, family = "binomial")
fit3 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl, data = noAD, family = "binomial")
fit4 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl + RAVLT_immediate_bl, data = noAD, family = "binomial")
fit5 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2, data = hasAP, family = "binomial")
fit6 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2 + PTEDUCAT, data = hasAP, family = "binomial")
fit7 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + ADASQ4_bl + RAVLT_immediate_bl + 
              APOE4_1 + APOE4_2 + AGE*RAVLT_immediate_bl*CollegePlus + AGE*MMSE_bl*CollegePlus , data = hasAP, family = "binomial")


# Predictions
pred1 <- predict(fit1, type = "response")
pred2 <- predict(fit2, type = "response")
pred3 <- predict(fit3, type = "response")
pred4 <- predict(fit4, type = "response")
pred5 <- predict(fit5, type = "response")
pred6 <- predict(fit6, type = "response")
pred7 <- predict(fit7, type = "response")
# True Labels
labs1 <- noAD$beta_pos_vote
labs2 <- noAD$beta_pos_vote
labs3 <- noAD$beta_pos_vote
labs4 <- noAD$beta_pos_vote
labs5 <- hasAP$beta_pos_vote
labs6 <- hasAP$beta_pos_vote
labs7 <- hasAP$beta_pos_vote
# Performance
perf1 <- performance(prediction(pred1, labs1), "tpr", "fpr")
perf2 <- performance(prediction(pred2, labs2), "tpr", "fpr")
perf3 <- performance(prediction(pred3, labs3), "tpr", "fpr")
perf4 <- performance(prediction(pred4, labs4), "tpr", "fpr")
perf5 <- performance(prediction(pred5, labs5), "tpr", "fpr")
perf6 <- performance(prediction(pred6, labs6), "tpr", "fpr")
perf7 <- performance(prediction(pred7, labs7), "tpr", "fpr")
# AUC
auc1 <- round(performance(prediction(pred1, labs1), measure = "auc")@y.values[[1]], 3)
auc2 <- round(performance(prediction(pred2, labs2), measure = "auc")@y.values[[1]], 3)
auc3 <- round(performance(prediction(pred3, labs3), measure = "auc")@y.values[[1]], 3)
auc4 <- round(performance(prediction(pred4, labs4), measure = "auc")@y.values[[1]], 3)
auc5 <- round(performance(prediction(pred5, labs5), measure = "auc")@y.values[[1]], 3)
auc6 <- round(performance(prediction(pred6, labs6), measure = "auc")@y.values[[1]], 3)
auc7 <- round(performance(prediction(pred7, labs7), measure = "auc")@y.values[[1]], 3)
# ROC Curve
c <- viridis::viridis(n = 7, begin = 0, end = 0.9)
plot(perf1, col = c[1], main = "Logistic Models")
plot(perf2, col = c[2], add = TRUE)
plot(perf3, col = c[3], add = TRUE)
plot(perf4, col = c[4], add = TRUE)
plot(perf5, col = c[5], add = TRUE)
plot(perf6, col = c[6], add = TRUE)
plot(perf7, col = c[7], add = TRUE)
legend(x = "bottomright", col = c, lty = 1, lwd = 2, cex = 0.45,
       title = c("Model Description & AUC"),
       ncol = 2, text.width = 0.35,
       legend = c(paste("Cog 1", auc1, sep = " - "),
                  paste("Cog 2", auc2, sep = " - "), 
                  paste("Cog 3", auc3, sep = " - "),
                  paste("Cog 4", auc4, sep = " - "),
                  paste("Cog 4 + Genetic", auc5, sep = " - "),
                  paste("Cog 4 + Genetic + Educ", auc6, sep = " - "),
                  paste("Cog 4 + Genetic + Age Interact", auc7, sep = " - ")))

aucs <- c(auc1, auc2, auc3, auc4, auc5, auc6, auc7)
cvs$auc <- aucs
cvs <- round(cvs, 3)

cvsp <- cvs %>% select(mis_class, true_pos, false_pos, auc, conv_corr, pct_pred_pos_conv, pct_pred_neg_conv)
colnames(cvsp) <- c("MisClass", "True P", "False P.", " AUC ", "Conv. Corr", "Pct Pos Conv.", "Pct Neg Conv.")
write_csv(cvsp, "processed_data/amyloid_cv_results.csv")

sample_cor <- round(cor(noAD$beta_pos_vote, noAD$AD_con_any), 3)

noAD_beta_pos <- noAD %>%
  filter(beta_pos_vote == 1)

noAD_beta_neg <- noAD %>%
  filter(beta_pos_vote == 0)

noAD_conv <- noAD %>%
  filter(AD_con_any == 1)

noAD_notconv <- noAD %>%
  filter(AD_con_any == 0)
sum(noAD_beta_neg$AD_con_any)
sample_pct_pos_conv <- round(sum(noAD_beta_pos$AD_con_any)/nrow(noAD_beta_pos), 3)
sample_pct_neg_conv <- round(sum(noAD_beta_neg$AD_con_any)/nrow(noAD_beta_neg), 3)

sample_ad_conv_pos <- round(sum(noAD_conv$beta_pos_vote)/nrow(noAD_conv), 3)
sample_ad_notconv_pos <- round(sum(noAD_notconv$beta_pos_vote)/nrow(noAD_notconv), 3)

cv_table <- ggtexttable(cvsp, 
                        theme = ttheme(
                          base_style = "lBlueWhite", 
                          rownames.style = rownames_style(face = "plain")))
cv_table <- tab_add_title(cv_table, text = "Stage 1 Models - 5 Fold CV Metrics",
                          hjust = -0.65)

sample_table <- data.frame(sample_cor, 
                           sample_pct_pos_conv, 
                           sample_pct_neg_conv,
                           sample_ad_conv_pos,
                           sample_ad_notconv_pos
                           )
colnames(sample_table) <- c("AD Conv Corr", 
                            "Pct Pos AD Conv", 
                            "Pct Neg AD Conv",
                            "Pct AD Conv Pos",
                            "Pct AD Not Conv Pos"
                            )
write_csv(sample_table, "processed_data/amyloid_sample_table.csv")

sample_table <- ggtexttable(sample_table,
                            rows = NULL,
                            theme = ttheme(
                              base_style = "lBlueWhite",
                              rownames.style = rownames_style(face = "plain")))
sample_table <- tab_add_title(sample_table, text = "Amyloid Sample Statistics",
                              hjust = -1.25)

ggsave("figures/amyloid_cv_table.pdf", cv_table, device = "pdf",
       width = 7, height = 5, units = "in")
ggsave("figures/amyloid_cv_table.png", cv_table, device = "png",
       width = 7, height = 5, units = "in")

ggsave("figures/amyloid_sample_table.pdf", sample_table, device = "pdf",
       width = 7.5, height = 4, units = "in")
ggsave("figures/amyloid_sample_table.png", sample_table, device = "png",
       width = 7.5, height = 4, units = "in")






