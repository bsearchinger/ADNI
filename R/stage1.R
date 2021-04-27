# Stage 1 - Amyloid Positivity

# Combine holdout set with RID
# push to github
# 

library(readr)
library(dplyr)
library(stringr)
library(origami)

amyloid_pos <- read_csv("processed_data/amyloid_pos_data.csv",
                        col_types = cols(av45_SUVR = col_double(),
                                         av45_pos = col_double(),
                                         AV45_bl = col_double(),
                                         br_vol = col_double())) 

adnim <- read_csv("processed_data/adnim.csv", 
                  col_types = cols(av45_SUVR = col_double(),
                                   av45_pos = col_double(),
                                   AV45_bl = col_double(),
                                   br_vol = col_double()))


holdout <- read_csv("processed_data/holdout.csv")

holdout_adnim <- adnim %>% filter(Holdout == 1)
holdout_adnim_bl <- holdout_adnim %>% filter(VISCODE == "bl")

non_holdout_pos <- amyloid_pos %>% filter(Holdout == 0, VISCODE == "bl", DX_bl != "AD")
non_holdout_pos <- non_holdout_pos[!is.na(non_holdout_pos$APOE4_1),]

model1 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + 
                ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2, 
              data = non_holdout_pos, family = "binomial")

model2 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + 
                ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2 + PTEDUCAT, 
              data = non_holdout_pos, family = "binomial")

model3 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + 
                ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2 + AGE*CollegePlus , 
              data = non_holdout_pos, family = "binomial")

pred1 <- predict(model1, newdata = holdout_adnim_bl, type = "response")
pred2 <- predict(model2, newdata = holdout_adnim_bl, type = "response")
pred3 <- predict(model3, newdata = holdout_adnim_bl, type = "response")

pred1 <- ifelse(pred1 > 0.5, 1, 0)
pred2 <- ifelse(pred2 > 0.5, 1, 0)
pred3 <- ifelse(pred3 > 0.5, 1, 0)
colnames(holdout_adnim_bl)
stage1_preds <- data.frame(PTID = holdout_adnim_bl$PTID,
                           RID = holdout_adnim_bl$RID,
                           VISCODE = holdout_adnim_bl$VISCODE,
                           DX_bl = holdout_adnim_bl$DX_bl,
                           AD_conv = holdout_adnim_bl$AD_con_any,
                           m1_pred = pred1,
                           m2_pred = pred2,
                           m3_pred = pred3)

write_csv(stage1_preds, "processed_data/state1_preds.csv")

table(stage1_preds$m1_pred, stage1_preds$DX_bl, stage1_preds$AD_conv)
table(stage1_preds$m2_pred, stage1_preds$DX_bl, stage1_preds$AD_conv)
table(stage1_preds$m3_pred, stage1_preds$DX_bl, stage1_preds$AD_conv)


