# Cleaning & Aggregating ADNI Data for Stage 1 Amyloid Model

# Required ADNI Datasets:
# ADNIMERGE
# UCBERKELEYAV45_01_14_21
# UPENNBIOMK_MASTER
# holdout.csv

require(tidyverse)

# Read In Data
adnim <- read_csv("ADNI_data/ADNIMERGE.csv") # Fix Column Types
upenn <- read_csv("ADNI_data/UPENNBIOMK_MASTER.csv")
av45 <- read_csv("ADNI_data/UCBERKELEYAV45_01_14_21.csv")
holdout <- read_csv("ConvImgs/holdout.csv")
hid <- unique(holdout$`Subject ID`)

# ADNIMERGE ####
# Relavent Columns of ADNIMERGE
adnim_cols <- c("RID", "PTID", "VISCODE", "AGE", "PTGENDER","PTEDUCAT", "APOE4", 
                "DX_bl", "DX",
                "FDG_bl", "FDG", 
                "AV45_bl", "AV45", 
                "ABETA_bl", "ABETA", 
                "TAU_bl", "TAU",
                "ADAS11_bl", "ADAS11",
                "ADAS13_bl", "ADAS13",
                "ADASQ4_bl", "ADASQ4",
                "LDELTOTAL_BL", "LDELTOTAL",
                "MMSE_bl", "MMSE",
                "RAVLT_immediate_bl", "RAVLT_immediate",
                "RAVLT_learning_bl", "RAVLT_learning",
                "RAVLT_forgetting_bl", "RAVLT_forgetting",
                "ICV_bl", "ICV")
# Subset Columns
adnim <- adnim[, adnim_cols]

# Create Dummy Variables for DX Conversion, Gender, Education, APOE, and Holdout
adnim <- adnim %>%
  mutate(AD_con = ifelse(DX_bl != "AD" & DX == "Dementia", 1, 0),
         CN_MCI = ifelse(DX_bl == "CN" & DX == "MCI", 1, 0),
         CD_AD = ifelse(DX_bl == "CN" & DX == "Dementia", 1, 0),
         MCI_AD = ifelse(DX_bl %in% c("EMCI", "LMCI") & DX == "Dementia", 1, 0),
         Male = ifelse(PTGENDER == "Male", 1, 0),
         NoHighSch = ifelse(PTEDUCAT < 12, 1, 0),
         HighSch = ifelse(PTEDUCAT == 12, 1, 0),
         SomeCollege = ifelse(PTEDUCAT > 12 & PTEDUCAT < 16, 1, 0),
         CollegePlus = ifelse(PTEDUCAT >= 16, 1, 0),
         APOE4_1 = ifelse(APOE4 == 1, 1, 0),
         APOE4_2 = ifelse(APOE4 == 2, 1, 0),
         Holdout = ifelse(PTID %in% hid, 1, 0))

# Conversion Data
ad_conv <- adnim %>% 
  filter(AD_con == 1) %>%
  select(RID)
adnim <- adnim %>%
  mutate(AD_con_any = ifelse(RID %in% unique(ad_conv$RID), 1, 0))

any_conv <- adnim %>%
  filter(AD_con == 1 | CN_MCI == 1) %>%
  select(RID)
adnim <- adnim %>%
  mutate(any_con = ifelse(RID %in% unique(any_conv$RID), 1, 0))

# Amyloid Positivity Data ####

# UPENN Median ABETA and Positivity By Patient and Visit Code
upenn_med <- upenn %>%
  filter(BATCH == "MEDIAN")
upenn_any_pos <- upenn_med %>%
  group_by(RID) %>%
  summarise(upenn_any_pos = any(ABETA < 192))

# Baseline Visit
upenn_pos_bl <- upenn_med %>%
  filter(VISCODE == "bl") %>%
  mutate(upenn_pos_bl = ifelse(ABETA < 192, 1, 0))
upenn_pos_bl = upenn_pos_bl[,c("RID", "upenn_pos_bl", "ABETA")]
colnames(upenn_pos_bl) <- c("RID", "upenn_pos_bl", "UPENN_ABETA_bl")
upenn_pos <- merge(upenn_any_pos, upenn_pos_bl, by = "RID", all.x = TRUE)
upenn_pos$upenn_any_pos <- ifelse(upenn_pos$upenn_any_pos, 1, 0)

# UC Berkeley AV45 SUVR
av45_bl <- av45 %>%
  filter(VISCODE == "bl")
av45_bl <- av45_bl[, c("RID", "SUMMARYSUVR_WHOLECEREBNORM", 
                       "SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF", 
                       "TOTAL_INTRACRANIAL_VOLUME")]
colnames(av45_bl) <- c("RID", "av45_SUVR", "av45_pos", "br_vol")

# UC Berkeley FDG
fdg <- read_csv("ADNI_data/UCBERKELEYFDG_05_28_20.csv")

# Merge Into ADNIMERGE
adnim <- merge(adnim, upenn_pos, by = c("RID"), all.x = TRUE)
adnim <- merge(adnim, av45_bl, by = c("RID"), all.x = TRUE)

# Positivity based on ABETA
adnim <- adnim %>%
  mutate(abeta_pos = ifelse(ABETA < 977, 1, 0))

# Filter Out People With No Beta Positivity Measures
adnim_pos <- adnim %>%
  filter(upenn_any_pos %in% c(0,1) | av45_pos %in% c(0,1) | abeta_pos %in% c(0,1))

# Code Discrepancies as Positive
beta_pos_vote <- adnim_pos %>% 
  dplyr::select(abeta_pos, av45_pos, upenn_any_pos) %>%
  rowSums(na.rm = TRUE)
adnim_pos$beta_pos_vote <- ifelse(beta_pos_vote > 0, 1, 0)

# Filter by baseline
#adnim_pos_bl <- adnim_pos %>% filter(VISCODE == "bl")
#adnim_pos_bl <- adnim_pos_bl[!is.na(adnim_pos_bl$DX_bl), ]

# Write Amyloid Positivity Data -- Dimensions: 10611 x 57
write_csv(adnim_pos, "processed_data/amyloid_pos_data.csv")
write_csv(adnim, "processed_data/adnim.csv")





