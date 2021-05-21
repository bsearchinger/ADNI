# MRI & PET Volumetric Analysis

library(tidyverse)
library(glmnet)
library(ROCR)
library(ggpubr)
library(assertthat)
# Read in Volumetric Data
MRI <- read_csv("processed_data/master_mri_volume.csv")
PETv <- read_csv("processed_data/master_pet_volume.csv")
PETs <- read_csv("processed_data/master_pet_suvr.csv")

# Holdout Data
holdout <- read_csv("processed_data/holdout.csv")

# Conversion Data
adnim <- read_csv("processed_data/adnim.csv",
                  col_types = cols(
                    AV45_bl = col_double(),
                    av45_SUVR_bl = col_double(),
                    br_vol_bl = col_double()))

adni_conv <- adnim %>%
  select(RID, AD_con_any, any_con, DX_bl)
adni_conv <- adni_conv[!duplicated(adni_conv),]

combined_v <- merge(MRI, PETv, by = c("RID", "VISCODE"))
combined_s <- merge(MRI, PETs, by = c("RID", "VISCODE"))

combined_v <- merge(combined_v, adni_conv, by = "RID", all.y = FALSE)
combined_s <- merge(combined_s, adni_conv, by = "RID", all.y = FALSE)

combined_v_nh <- combined_v %>% filter(RID %in% holdout$RID == F)
combined_v_h <- combined_v %>% filter(RID %in% holdout$RID == T)

combined_s_nh <- combined_s %>% filter(RID %in% holdout$RID == F)
combined_s_h <- combined_s %>% filter(RID %in% holdout$RID == T)

# Check for no holdout data
assert_that(any(holdout$RID %in% combined_v_nh$RID) == F,
            any(holdout$RID %in% combined_s_nh$RID) == F,
            msg = "Holdout Cases Detected in Training Data")

# Indicator for AD at baseline
combined_v_nh$AD_bl <- ifelse(combined_v_nh$DX_bl == "AD", 1, 0) 
combined_s_nh$AD_bl <- ifelse(combined_s_nh$DX_bl == "AD", 1, 0) 

combined_v_h$AD_bl <- ifelse(combined_v_h$DX_bl == "AD", 1, 0) 
combined_s_h$AD_bl <- ifelse(combined_s_h$DX_bl == "AD", 1, 0) 

# Indicator for AD at baseline or LMCI at baseline and eventual convert to AD
combined_v_nh$AD_LMCI_conv <- ifelse(
  combined_v_nh$DX_bl == "AD" | 
    (combined_v_nh$DX_bl == "LMCI" & combined_v_nh$AD_con_any == 1), 1, 0)
combined_s_nh$AD_LMCI_conv <- ifelse(
  combined_s_nh$DX_bl == "AD" | 
    (combined_s_nh$DX_bl == "LMCI" & combined_s_nh$AD_con_any == 1), 1, 0)

combined_v_h$AD_LMCI_conv <- ifelse(
  combined_v_h$DX_bl == "AD" | 
    (combined_v_h$DX_bl == "LMCI" & combined_v_h$AD_con_any == 1), 1, 0)
combined_s_h$AD_LMCI_conv <- ifelse(
  combined_s_h$DX_bl == "AD" | 
    (combined_s_h$DX_bl == "LMCI" & combined_s_h$AD_con_any == 1), 1, 0)

combined_v_nh <- na.omit(combined_v_nh)
combined_s_nh <- na.omit(combined_s_nh)

combined_v_h <- na.omit(combined_v_h)
combined_s_h <- na.omit(combined_s_h)

write_csv(combined_s_h, "processed_data/volume_suvr_holdout.csv")


# Training Set
training_set_v <- combined_v_nh %>% 
  filter(DX_bl %in% c("AD", "CN")) %>%
  mutate(AD = ifelse(DX_bl == "AD", 1, 0))
training_set_s <- combined_s_nh %>% 
  filter(DX_bl %in% c("AD", "CN")) %>%
  mutate(AD = ifelse(DX_bl == "AD", 1, 0))

# Validation Sets
set.seed(12345)
# Volume
scms_v <- combined_v_nh %>% 
  filter(DX_bl == "SMC") %>%
  mutate(AD = 0)
scm_v_vals <- sample(1:nrow(scms_v), 15)
scm_v_val <- scms_v[scm_v_vals, ]

combined_v_vals <- combined_v_nh %>% 
  filter(DX_bl %in% c("EMCI", "LMCI")) %>%
  mutate(AD = ifelse(AD_con_any == 1, 1, 0))

validation_ad_v <- combined_v_vals %>% filter(AD == 1)
validation_cn_v <- combined_v_vals %>% filter(AD == 0)
validation_sample <- sample(1:nrow(validation_cn_v), 30)
validation_cn_v <- validation_cn_v[validation_sample, ]
validation_cn_v <- rbind(validation_cn_v, scm_v_val)
validation_set_v <- rbind(validation_ad_v, validation_cn_v)

# SUVR
scms_s <- combined_s_nh %>% 
  filter(DX_bl == "SMC") %>%
  mutate(AD = 0)
scm_s_val <- scms_s[scm_v_vals, ]

combined_s_vals <- combined_s_nh %>% 
  filter(DX_bl %in% c("EMCI", "LMCI")) %>%
  mutate(AD = ifelse(AD_con_any == 1, 1, 0))

validation_ad_s <- combined_s_vals %>% filter(AD == 1)
validation_cn_s <- combined_s_vals %>% filter(AD == 0)
validation_cn_s <- validation_cn_s[validation_sample, ]
validation_cn_s <- rbind(validation_cn_s, scm_s_val)
validation_set_s <- rbind(validation_ad_s, validation_cn_s)

# Fit Models Here
mm_v_train <- model.matrix(AD ~ ., training_set_v[,c(3:155, 161)])
mm_v_val <- as.matrix(validation_set_v[, c(3:155)])
mm_v_val <- cbind(rep(1, nrow(mm_v_val)), mm_v_val)

mm_s_train <- model.matrix(AD ~ ., training_set_s[,c(3:155, 161)])
mm_s_val <- as.matrix(validation_set_s[, c(3:155)])
mm_s_val <- cbind(rep(1, nrow(mm_s_val)), mm_s_val)


# Volume Train
elastic_net_v_train <- lapply(seq(0, 1, by = 0.1), function(a) {
  fit <- cv.glmnet(x = mm_v_train, y = training_set_v$AD, alpha = a, family = "binomial")
  lambda <- fit$lambda.min
  pred_vals <- predict(fit, newx = mm_v_train, s = lambda, type = "response")
  pred_nz <- predict(fit, newx = mm_v_train, s = lambda, type = "nonzero")
  pred_coef <- as.matrix(predict(fit, newx = mm_v_train, s = lambda, type = "coefficients"))
  pred_coef <- pred_coef[pred_nz$X1,]
  
  perf <- performance(prediction(pred_vals, training_set_v$AD), "tpr", "fpr")
  auc <- round(performance(prediction(pred_vals, training_set_v$AD), measure = "auc")@y.values[[1]], 3)
  
  out = list(alpha = a,
             pred_vals = pred_vals,
             lambda = lambda,
             coef = pred_coef,
             perf = perf,
             auc = auc
             )
  return(out)
})
# Volume Validation
elastic_net_v_val <- lapply(seq(0, 1, by = 0.1), function(a) {
  fit <- cv.glmnet(x = mm_v_train, y = training_set_v$AD, alpha = a, family = "binomial")
  lambda <- fit$lambda.min
  pred_vals <- predict(fit, newx = mm_v_val, s = lambda, type = "response")
  pred_nz <- predict(fit, newx = mm_v_val, s = lambda, type = "nonzero")
  pred_coef <- as.matrix(predict(fit, newx = mm_v_val, s = lambda, type = "coefficients"))
  pred_coef <- pred_coef[pred_nz$X1,]
  
  perf <- performance(prediction(pred_vals, validation_set_v$AD), "tpr", "fpr")
  auc <- round(performance(prediction(pred_vals, validation_set_v$AD), measure = "auc")@y.values[[1]], 3)
  
  out = list(alpha = a,
             pred_vals = pred_vals,
             lambda = lambda,
             coef = pred_coef,
             perf = perf,
             auc = auc
  )
  return(out)
})

# SUVR Train
elastic_net_s_train <- lapply(seq(0, 1, by = 0.1), function(a) {
  fit <- cv.glmnet(x = mm_s_train, y = training_set_s$AD, alpha = a, family = "binomial")
  lambda <- fit$lambda.min
  pred_vals <- predict(fit, newx = mm_s_train, s = lambda, type = "response")
  pred_nz <- predict(fit, newx = mm_s_train, s = lambda, type = "nonzero")
  pred_coef <- as.matrix(predict(fit, newx = mm_s_train, s = lambda, type = "coefficients"))
  pred_coef <- pred_coef[pred_nz$X1,]
  
  perf <- performance(prediction(pred_vals, training_set_s$AD), "tpr", "fpr")
  auc <- round(performance(prediction(pred_vals, training_set_s$AD), measure = "auc")@y.values[[1]], 3)
  
  out = list(alpha = a,
             pred_vals = pred_vals,
             lambda = lambda,
             coef = pred_coef,
             perf = perf,
             auc = auc,
             model = fit
  )
  return(out)
})
# SUVR Validation
elastic_net_s_val <- lapply(seq(0, 1, by = 0.1), function(a) {
  fit <- cv.glmnet(x = mm_s_train, y = training_set_s$AD, alpha = a, family = "binomial")
  lambda <- fit$lambda.min
  pred_vals <- predict(fit, newx = mm_s_val, s = lambda, type = "response")
  pred_nz <- predict(fit, newx = mm_s_val, s = lambda, type = "nonzero")
  pred_coef <- as.matrix(predict(fit, newx = mm_s_val, s = lambda, type = "coefficients"))
  pred_coef <- pred_coef[pred_nz$X1,]
  
  perf <- performance(prediction(pred_vals, validation_set_s$AD), "tpr", "fpr")
  auc <- round(performance(prediction(pred_vals, validation_set_s$AD), measure = "auc")@y.values[[1]], 3)
  
  out = list(alpha = a,
             pred_vals = pred_vals,
             lambda = lambda,
             coef = pred_coef,
             perf = perf,
             auc = auc
  )
  return(out)
})

alpha_n <- paste0("alpha", stringr::str_pad(0:10, width = 3, pad = "0"))
names(elastic_net_v_train) <- alpha_n
names(elastic_net_v_val) <- alpha_n
names(elastic_net_s_train) <- alpha_n
names(elastic_net_s_val) <- alpha_n

# ROC Curves
c <- viridis::viridis(n = 11, begin = 0, end = 0.9)
plot(elastic_net_v_train[[1]]$perf, col = c[1], main = "Elastic Net Volume Models - Training Set")
plot(elastic_net_v_train[[2]]$perf, col = c[2], add = TRUE)
plot(elastic_net_v_train[[3]]$perf, col = c[3], add = TRUE)
plot(elastic_net_v_train[[4]]$perf, col = c[4], add = TRUE)
plot(elastic_net_v_train[[5]]$perf, col = c[5], add = TRUE)
plot(elastic_net_v_train[[6]]$perf, col = c[6], add = TRUE)
plot(elastic_net_v_train[[7]]$perf, col = c[7], add = TRUE)
plot(elastic_net_v_train[[8]]$perf, col = c[8], add = TRUE)
plot(elastic_net_v_train[[9]]$perf, col = c[9], add = TRUE)
plot(elastic_net_v_train[[10]]$perf, col = c[10], add = TRUE)
plot(elastic_net_v_train[[11]]$perf, col = c[11], add = TRUE)


suvr_val_preds <- elastic_net_s_val$alpha000$pred_vals[,1]
suvr_val_truth <- validation_set_s$AD_con_any
suvr_val_preds <- data.frame(truth = suvr_val_truth,
                             preds = suvr_val_preds)
write_csv(suvr_val_preds, "processed_data/suvr_val_preds.csv")
suvr_val_preds <- suvr_val_preds %>%
  mutate(c25 = ifelse(preds > 0.25, 1, 0),
         c50 = ifelse(preds > 0.50, 1, 0),
         c75 = ifelse(preds > 0.75, 1, 0))

table(suvr_val_preds$truth, suvr_val_preds$c25)
table(suvr_val_preds$truth, suvr_val_preds$c50)
table(suvr_val_preds$truth, suvr_val_preds$c75)

sum(diag(table(suvr_val_preds$truth, suvr_val_preds$c25)))/nrow(suvr_val_preds)
sum(diag(table(suvr_val_preds$truth, suvr_val_preds$c50)))/nrow(suvr_val_preds)
sum(diag(table(suvr_val_preds$truth, suvr_val_preds$c75)))/nrow(suvr_val_preds)

auc_df <- data.frame(
  alpha = seq(0, 1, by = 0.1), 
 auc_volume_train = c(elastic_net_v_train[[1]]$auc,
                      elastic_net_v_train[[2]]$auc,
                      elastic_net_v_train[[3]]$auc,
                      elastic_net_v_train[[4]]$auc,
                      elastic_net_v_train[[5]]$auc,
                      elastic_net_v_train[[6]]$auc,
                      elastic_net_v_train[[7]]$auc,
                      elastic_net_v_train[[8]]$auc,
                      elastic_net_v_train[[9]]$auc,
                      elastic_net_v_train[[10]]$auc,
                      elastic_net_v_train[[11]]$auc),
 auc_volume_val = c(elastic_net_v_val[[1]]$auc,
                    elastic_net_v_val[[2]]$auc,
                    elastic_net_v_val[[3]]$auc,
                    elastic_net_v_val[[4]]$auc,
                    elastic_net_v_val[[5]]$auc,
                    elastic_net_v_val[[6]]$auc,
                    elastic_net_v_val[[7]]$auc,
                    elastic_net_v_val[[8]]$auc,
                    elastic_net_v_val[[9]]$auc,
                    elastic_net_v_val[[10]]$auc,
                    elastic_net_v_val[[11]]$auc),
 auc_suvr_train = c(elastic_net_s_train[[1]]$auc,
                      elastic_net_s_train[[2]]$auc,
                      elastic_net_s_train[[3]]$auc,
                      elastic_net_s_train[[4]]$auc,
                      elastic_net_s_train[[5]]$auc,
                      elastic_net_s_train[[6]]$auc,
                      elastic_net_s_train[[7]]$auc,
                      elastic_net_s_train[[8]]$auc,
                      elastic_net_s_train[[9]]$auc,
                      elastic_net_s_train[[10]]$auc,
                      elastic_net_s_train[[11]]$auc),
 auc_suvr_val = c(elastic_net_s_val[[1]]$auc,
                    elastic_net_s_val[[2]]$auc,
                    elastic_net_s_val[[3]]$auc,
                    elastic_net_s_val[[4]]$auc,
                    elastic_net_s_val[[5]]$auc,
                    elastic_net_s_val[[6]]$auc,
                    elastic_net_s_val[[7]]$auc,
                    elastic_net_s_val[[8]]$auc,
                    elastic_net_s_val[[9]]$auc,
                    elastic_net_s_val[[10]]$auc,
                    elastic_net_s_val[[11]]$auc))

write_csv(auc_df, "processed_data/volumetric_auc.csv")
colnames(auc_df) <- c("alpha", "Volume Train", "Volume Validation", "SUVR Train", "SUVR Validation")
auc_gg <- ggtexttable(auc_df, rows = NULL,
                      theme = ttheme(
                        base_style = "lBlueWhite",
                        rownames.style = rownames_style(face = "plain")))
auc_gg <- tab_add_title(auc_gg, text = "Volume vs. SUVR Elastic Net AUC",
                        hjust = -0.5)
ggsave("figures/elastic_net_auc.pdf", plot = auc_gg, device = "pdf", 
       width = 7, height = 4.75, units = "in")
ggsave("figures/elastic_net_auc.png", plot = auc_gg, device = "png", 
       width = 7, height = 4.75, units = "in", dpi = 300)

# Best fit is ridge regression with MRI volume + PET SUVR
final_set <- rbind(training_set_s, validation_set_s)
final_set_mm <- model.matrix(AD ~ ., final_set[,c(3:155, 161)])

final_fit <- glmnet(x = final_set_mm, y = final_set$AD, alpha = 0, 
                    family = "binomial", lambda = elastic_net_s_val[[1]]$lambda)

final_preds <- predict(final_fit, newx = final_set_mm, type = "response") 
final_perf <- performance(prediction(final_preds, final_set$AD), "tpr", "fpr")
final_auc <- round(performance(prediction(final_preds, final_set$AD), measure = "auc")@y.values[[1]], 3)

final_lambda <- elastic_net_s_val[[1]]$lambda
write_csv(final_set, paste("processed_data/final_set", final_lambda, ".csv", sep = ""))

# Plot Final vs Validation
plot(final_perf, col = c[1], main = "Ridge Regression", lwd = 1.5, 
     sub = paste("Lambda =", round(elastic_net_s_val[[1]]$lambda, 5)))
plot(elastic_net_s_val[[1]]$perf, col = c[10], lwd = 1.5, add = TRUE)
legend(x = 0.4, y = 0.2, col = c(c[1], c[10]), lty = 1, lwd = 2, cex = 1,
       legend = c(paste("Final Model: ", final_auc, " AUC", sep = ""),
                  paste("Validation Model: ", elastic_net_s_val[[1]]$auc, " AUC", sep = "")))


# Final Set should have 236 "CN" and 93 "AD"

# ROC Curve
# c <- viridis::viridis(n = 11, begin = 0, end = 0.9)
# plot(elastic_net_val[[1]]$perf, col = c[1], main = "Elastic Net Models")
# plot(elastic_net_val[[2]]$perf, col = c[2], add = TRUE)
# plot(elastic_net_val[[3]]$perf, col = c[3], add = TRUE)
# plot(elastic_net_val[[4]]$perf, col = c[4], add = TRUE)
# plot(elastic_net_val[[5]]$perf, col = c[5], add = TRUE)
# plot(elastic_net_val[[6]]$perf, col = c[6], add = TRUE)
# plot(elastic_net_val[[7]]$perf, col = c[7], add = TRUE)
# plot(elastic_net_val[[8]]$perf, col = c[8], add = TRUE)
# plot(elastic_net_val[[9]]$perf, col = c[9], add = TRUE)
# plot(elastic_net_val[[10]]$perf, col = c[10], add = TRUE)
# plot(elastic_net_val[[11]]$perf, col = c[11], add = TRUE)




