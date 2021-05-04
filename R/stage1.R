# Stage 1 - Amyloid Positivity

# Combine holdout set with RID
# push to github
# 

library(readr)
library(dplyr)
library(stringr)
library(origami)

amyloid_pos <- read_csv("processed_data/amyloid_pos_data2.csv",
                        col_types = cols(av45_SUVR_bl = col_double(),
                                         av45_any_pos = col_double(),
                                         AV45_bl = col_double(),
                                         br_vol_bl = col_double())) 

adnim <- read_csv("processed_data/adnim2.csv", 
                  col_types = cols(av45_SUVR_bl = col_double(),
                                   av45_any_pos = col_double(),
                                   AV45_bl = col_double(),
                                   br_vol_bl = col_double()))


holdout <- read_csv("processed_data/holdout.csv")

holdout_adnim <- adnim %>% filter(Holdout == 1)
holdout_adnim_bl <- holdout_adnim %>% filter(VISCODE == "bl")

non_holdout_pos <- amyloid_pos %>% filter(Holdout == 0, VISCODE == "bl", DX_bl != "AD")
non_holdout_pos <- non_holdout_pos[!is.na(non_holdout_pos$APOE4_1),]

holdout_pos_bl <- amyloid_pos %>% filter(Holdout == 1, VISCODE == "bl")

model1 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + 
                ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2, 
              data = non_holdout_pos, family = "binomial")

model2 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + 
                ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2 + PTEDUCAT, 
              data = non_holdout_pos, family = "binomial")

model3 <- glm(beta_pos_vote ~ AGE + Male + MMSE_bl + LDELTOTAL_BL + 
                ADASQ4_bl + RAVLT_immediate_bl + APOE4_1 + APOE4_2 + 
                AGE*RAVLT_immediate_bl*CollegePlus + AGE*MMSE_bl*CollegePlus, 
              data = non_holdout_pos, family = "binomial")

pred1 <- predict(model1, newdata = holdout_pos_bl, type = "response")
pred2 <- predict(model2, newdata = holdout_pos_bl, type = "response")
pred3 <- predict(model3, newdata = holdout_pos_bl, type = "response")


preds1 <- ifelse(pred1 > 0.5, 1, 0)
preds2 <- ifelse(pred2 > 0.5, 1, 0)
preds3 <- ifelse(pred3 > 0.5, 1, 0)

stage1_preds <- data.frame(PTID = holdout_pos_bl$PTID,
                           RID = holdout_pos_bl$RID,
                           VISCODE = holdout_pos_bl$VISCODE,
                           DX_bl = holdout_pos_bl$DX_bl,
                           amyloid_pos = holdout_pos_bl$beta_pos_vote,
                           AD_conv = holdout_pos_bl$AD_con_any,
                           m1_response = pred1,
                           m2_response = pred2,
                           m3_response = pred3,
                           m1_0.5_pred = preds1,
                           m2_0.5_pred = preds2,
                           m3_0.5_pred = preds3)

t <- table(holdout_pos_bl$AD_con_any, holdout_pos_bl$beta_pos_vote)
# In holdout set, 8 out of 8 Ad coversions are amyloid positive
# and 11 out of 28 non-AD conversions are amyloid positive

write_csv(stage1_preds, "processed_data/stage1_preds.csv")

# holdout_adnim_na_bl_upen <- holdout_adnim %>% filter(RID %in% stage1_preds$RID[n]) %>%
#   select(RID, upenn_any_pos) %>% na.omit()
# holdout_adnim_na_bl_abeta <- holdout_adnim %>% filter(RID %in% stage1_preds$RID[n]) %>%
#   select(RID, abeta_pos) %>% na.omit()
# 
# no_betas <- merge(holdout_adnim_na_bl_upen, holdout_adnim_na_bl_abeta, by = "RID") %>%
#   group_by(RID) %>%
#   mutate(beta_pos_vote = ifelse(any(upenn_any_pos == 1) | any(abeta_pos == 1), 1, 0)) %>%
#   select(RID, beta_pos_vote)
# 
# no_betas <- no_betas[!duplicated(no_betas), ]
# 
# 
# 
# table(stage1_preds$m1_0.5_pred, stage1_preds$amyloid_pos)
# table(stage1_preds$m2_pred, stage1_preds$DX_bl, stage1_preds$AD_conv)
# table(stage1_preds$m3_pred, stage1_preds$AD_conv)
# 
# n <- which(is.na(stage1_preds$amyloid_pos))
# stage1_preds_nan <- stage1_preds[-c(n), ]
# table(stage1_preds_nan$m3_0.5_pred, stage1_preds_nan$amyloid_pos)
