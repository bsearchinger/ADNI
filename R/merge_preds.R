# Merge Predictions

library(readr)
library(dplyr)
# Read in All Prediction data sets
stage1_preds <- read_csv("processed_data/stage1_final_preds.csv") %>% 
  select(PTID, DX_bl, stage1_response, stage1_pred, amyloid_pos)
ensemble_preds <- read_csv("processed_data/MRI_PET_Ensemble_Preds.csv") %>%
  select(PTID, MRI_Score, PET_Score, MRI_PET_Ensemble, AD_Conv)
#ridge_preds <- read_csv("processed_data/volume_holdout_preds.csv")

final_preds <- merge(stage1_preds, ensemble_preds, by = "PTID")
write_csv(final_preds, "processed_data/final_preds.csv")

final_table <- final_preds[,-1]
colnames(final_table) <- c("Baseline DX", "Stage 1 Probability", "Stage 1 Prediction",
                           "Amyloid Positive", "MRI Score", "PET Score", "Ensemble Prediction",
                           "AD Conversion")

final_table <- ggtexttable(final_table, 
                            rows = NULL,
                            theme = ttheme(
                              base_style = "lBlueWhite",
                              rownames.style = rownames_style(
                                face = "plain")))
ggsave("figures/final_table.pdf", plot = final_table,
       device = "pdf", width = 11, height = 11, units = "in")


# table(final_preds$AD_conv, final_preds$`PET CNN Prediction`)
# 
# final_preds_m1_0 <- final_preds %>% 
#   filter(m1_0.5_pred == 0)
# 
# final_preds_m1_1 <- final_preds %>% 
#   filter(m1_0.5_pred == 1)
# 
# final_preds_m2_0 <- final_preds %>% 
#   filter(m2_pred == 0)
# 
# final_preds_m2_1 <- final_preds %>% 
#   filter(m2_pred == 1)
# 
# final_preds_m3_0 <- final_preds %>% 
#   filter(m3_pred == 0)
# 
# final_preds_m3_1 <- final_preds %>% 
#   filter(m3_pred == 1)
# 
# table(final_preds_m1_0$AD_conv, final_preds_m1_0$`PET CNN Prediction`)
# table(final_preds_m1_1$AD_conv, final_preds_m1_1$`PET CNN Prediction`)
# 
# table(final_preds_m1_0$AD_conv, final_preds_m1_0$ridge_gt0.15)
# table(final_preds_m1_1$AD_conv, final_preds_m1_1$ridge_gt0.15)
# 
# table(final_preds_m2_0$AD_conv, final_preds_m2_0$`PET CNN Prediction`)
# table(final_preds_m2_1$AD_conv, final_preds_m2_1$`PET CNN Prediction`)
# 
# table(final_preds_m3_0$AD_conv, final_preds_m3_0$`PET CNN Prediction`)
# table(final_preds_m3_1$AD_conv, final_preds_m3_1$`PET CNN Prediction`)
# 
# 
# table(final_preds$AD_conv, final_preds$m1_pred)
# table(final_preds$AD_conv, final_preds$m2_pred)
# table(final_preds$AD_conv, final_preds$m3_pred)



