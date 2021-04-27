# ADNI MRI Volume Functions

# ADNI Files Used
# ASHS Volume Data [ADNI3]
# UCD White Matter Hyperintensity Volumes [ADNI1, GO, 2, 3]
# ADNIMERGE

library(tidyverse)

# Read in Data and Helper Funs
ashs <- read_csv("ADNI_data/ADNI_PICSLASHS_05_05_20.csv")
whm <- read_csv("ADNI_data/ADNI_UCD_WMH_09_01_20.csv")
adni <- read_csv("ADNI_data/ADNIMERGE.csv")
source("R/helper_funs.R")

# ASHS
ashs <- dplyr::select(ashs, RID, VISCODE2, ICV, dplyr::ends_with("VOL"))

# White Matter
whm <- dplyr::select(whm, RID, VISCODE2, CEREBRUM_TCV:TOTAL_BRAIN)
whm <- na.omit(whm)

# Merge Volume Data, Change VISCODE2 Name and Recode Screening to Baseline
master_vol <- merge(ashs, whm, by = c("RID", "VISCODE2"), all = FALSE)
master_vol <- dplyr::rename(master_vol, VISCODE = VISCODE2)
master_vol$VISCODE[master_vol$VISCODE == "sc"] <- "bl"

# Get DX Conversions and Merge
#adni <- adniConvert(adni)
#master_vol <- merge(master_vol, adni, by = c("RID", "VISCODE"), all = FALSE)
readr::write_csv(master_vol, "processed_data/master_mri_volume.csv")

# Final Data Dimensions: 1127 x 47





