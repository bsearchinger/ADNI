library(tidyverse)
library(glmnet)

# read in final training data and holdout data
final <- read_csv("processed_data/final_set0.0518844239876985.csv")
volume_holdout <- read_csv("processed_data/volume_suvr_holdout.csv")

# Fit Model
final_mm <- model.matrix(AD ~ ., data = final[,c(3:155, 161)])
lambda <- 0.0518844239876985
final_fit <- glmnet(x = final_mm, y = final$AD, alpha = 0, lambda = lambda, family = "binomial")

# Predict Holdout
holdout_mm <- model.matrix(AD_con_any ~ ., data = volume_holdout[, c(3:156)])
holdout_preds <- predict(final_fit, newx = holdout_mm, type = "response")

perf <- performance(prediction(holdout_preds, volume_holdout$AD_con_any), "tpr", "fpr")
auc <- round(performance(prediction(holdout_preds, volume_holdout$AD_con_any), measure = "auc")@y.values[[1]], 3)
# AUC 0.679

volume_holdout$ridge_response <- holdout_preds
volume_holdout_preds <- volume_holdout %>%
  select(RID, DX_bl, VISCODE, AD_con_any, ridge_response) %>%
  mutate(ridge_gt0.15 = ifelse(ridge_response > 0.15, 1, 0),
         ridge_gt0.25 = ifelse(ridge_response > 0.25, 1, 0),
         ridge_gt0.5  = ifelse(ridge_response > 0.5, 1, 0)) 

write_csv(volume_holdout_preds, "processed_data/volume_holdout_preds.csv")


