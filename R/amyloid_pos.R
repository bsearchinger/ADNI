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
holdout <- read_csv("processed_data/holdout.csv") %>% mutate(PTID = `Subject ID`)
hid <- unique(holdout$PTID)

# ADNIMERGE ####
# Relevant Columns of ADNIMERGE
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
upenn_pos_bl <- upenn_pos_bl %>% 
  select(RID, upenn_pos_bl, ABETA) %>%
  rename(UPENN_ABETA_bl = ABETA)
  
upenn_pos <- merge(upenn_any_pos, upenn_pos_bl, by = "RID", all.x = TRUE)
upenn_pos$upenn_any_pos <- ifelse(upenn_pos$upenn_any_pos, 1, 0)

# UC Berkeley AV45 SUVR
first_visit <- av45 %>% 
  group_by(RID) %>%
  arrange(EXAMDATE, .by_group = TRUE) %>% 
  summarise(first_vis = first(VISCODE2),
            first_date = first(EXAMDATE))

first_vis_pos <- rep(NA, nrow(first_visit))
for (i in 1:nrow(first_visit)) {
  fv <- av45 %>% 
    filter(RID == first_visit$RID[i] & VISCODE2 == first_visit$first_vis[i])
  first_vis_pos[i] <- ifelse(fv$SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF[1] == 1, 1, 0)
}

first_vis_pos <- data.frame(RID = first_visit$RID, first_vis_pos = first_vis_pos)

av45_any_pos <- av45 %>%
  group_by(RID) %>%
  summarise(av45_any_pos = any(SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF == 1)) %>%
  mutate(av45_any_pos = ifelse(av45_any_pos, 1, 0))

av45_bl <- av45 %>%
  filter(VISCODE == "bl") %>%
  select(RID, SUMMARYSUVR_WHOLECEREBNORM,
         SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF,
         TOTAL_INTRACRANIAL_VOLUME) %>%
  rename(av45_SUVR_bl = SUMMARYSUVR_WHOLECEREBNORM,
         av45_pos_bl = SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF,
         br_vol_bl = TOTAL_INTRACRANIAL_VOLUME)

av45_pos <- merge(av45_any_pos, first_vis_pos, by = "RID", all.x = TRUE)
av45_pos <- merge(av45_pos, av45_bl, by = "RID", all.x = TRUE)

# UC Berkeley FDG
#fdg <- read_csv("ADNI_data/UCBERKELEYFDG_05_28_20.csv")

# Merge Into ADNIMERGE
adnim <- merge(adnim, upenn_pos, by = c("RID"), all.x = TRUE)
adnim <- merge(adnim, av45_pos, by = c("RID"), all.x = TRUE)

# Positivity based on ABETA
adnim <- adnim %>%
  mutate(abeta_pos = ifelse(ABETA < 977, 1, 0))

# Filter Out People With No Beta Positivity Measures
adnim_pos <- adnim %>%
  filter(upenn_any_pos %in% c(0,1) | av45_any_pos %in% c(0,1) | abeta_pos %in% c(0,1))

# Code Discrepancies as Positive
beta_pos_vote <- adnim_pos %>% 
  dplyr::select(abeta_pos, av45_any_pos, upenn_any_pos) %>%
  rowSums(na.rm = TRUE)
adnim_pos$beta_pos_vote <- ifelse(beta_pos_vote > 0, 1, 0)


# Filter by baseline
#adnim_pos_bl <- adnim_pos %>% filter(VISCODE == "bl")
#adnim_pos_bl <- adnim_pos_bl[!is.na(adnim_pos_bl$DX_bl), ]

### OLD ###
# Write Amyloid Positivity Data -- Dimensions: 10611 x 57
#write_csv(adnim_pos, "processed_data/amyloid_pos_data.csv")
#write_csv(adnim, "processed_data/adnim.csv")

# NEW ###
# Write Amyloid Positivity Data -- Dimensions: 12330 x 57
write_csv(adnim_pos, "processed_data/amyloid_pos_data.csv")
write_csv(adnim, "processed_data/adnim.csv")


