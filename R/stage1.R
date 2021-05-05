# Stage 1 - Amyloid Positivity

# Combine holdout set with RID
# push to github
# 

library(readr)
library(dplyr)
library(stringr)
library(origami)
library(ggpubr)

amyloid_pos <- read_csv("processed_data/amyloid_pos_data.csv",
                        col_types = cols(av45_SUVR_bl = col_double(),
                                         av45_any_pos = col_double(),
                                         AV45_bl = col_double(),
                                         br_vol_bl = col_double())) 

adnim <- read_csv("processed_data/adnim.csv", 
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


stage1_preds <- data.frame(DX_bl = holdout_pos_bl$DX_bl,
                           amyloid_pos = holdout_pos_bl$beta_pos_vote,
                           response = pred1,
                           AD_conv = holdout_pos_bl$AD_con_any)

stage1_final_preds <- data.frame(PTID = holdout_pos_bl$PTID,
                                 RID = holdout_pos_bl$RID,
                                 VISCODE = holdout_pos_bl$VISCODE,
                                 DX_bl = holdout_pos_bl$DX_bl,
                                 amyloid_pos = holdout_pos_bl$beta_pos_vote,
                                 stage1_response = pred1,
                                 stage1_pred = ifelse(pred1 > 0.4, 1, 0))

write_csv(stage1_final_preds, "processed_data/stage1_final_preds.csv")

colnames(stage1_preds) <- c("Baseline DX", "Amyloid Positive", "Predicted Probability", "AD Conversion")
stage1_table <- ggtexttable(stage1_preds, 
                            rows = NULL,
                            theme = ttheme(
                              base_style = "lBlueWhite",
                              rownames.style = rownames_style(
                                face = "plain")))
ggsave("figures/stage1_table.pdf", plot = stage1_table,
       device = "pdf", width = 6, height = 11, units = "in")
ggsave("figures/stage1_table.png", plot = stage1_table,
       device = "png", width = 6, height = 11, units = "in")

