# Merge Predictions

library(readr)
library(dplyr)
stage1_preds <- read_csv("processed_data/stage1_preds.csv")
pet_preds <- read_csv("processed_data/holdout_preds_with_PET_ensemble.csv")
ridge_preds <- read_csv("processed_data/volume_holdout_preds.csv")

final_preds <- merge(pet_preds[, c(21,24, 25)], stage_1_preds[, c(2,4,5,7:12)], by = "RID")
final_preds <- merge(final_preds, ridge_preds[, c(1, 5:8)], by = "RID")


table(final_preds$AD_conv, final_preds$`PET CNN Prediction`)

final_preds_m1_0 <- final_preds %>% 
  filter(m1_pred == 0)

final_preds_m1_1 <- final_preds %>% 
  filter(m1_pred == 1)

final_preds_m2_0 <- final_preds %>% 
  filter(m2_pred == 0)

final_preds_m2_1 <- final_preds %>% 
  filter(m2_pred == 1)

final_preds_m3_0 <- final_preds %>% 
  filter(m3_pred == 0)

final_preds_m3_1 <- final_preds %>% 
  filter(m3_pred == 1)

table(final_preds_m1_0$AD_conv, final_preds_m1_0$`PET CNN Prediction`)
table(final_preds_m1_1$AD_conv, final_preds_m1_1$`PET CNN Prediction`)

table(final_preds_m2_0$AD_conv, final_preds_m2_0$`PET CNN Prediction`)
table(final_preds_m2_1$AD_conv, final_preds_m2_1$`PET CNN Prediction`)

table(final_preds_m3_0$AD_conv, final_preds_m3_0$`PET CNN Prediction`)
table(final_preds_m3_1$AD_conv, final_preds_m3_1$`PET CNN Prediction`)


table(final_preds$AD_conv, final_preds$m1_pred)
table(final_preds$AD_conv, final_preds$m2_pred)
table(final_preds$AD_conv, final_preds$m3_pred)



