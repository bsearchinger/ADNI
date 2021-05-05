# ADNI PET Volume Functions

# ADNI Files Used
# UC Berkeley - AV1451 Analysis [ADNI2, 3]
# ADNIMERGE

library(tidyverse)

# Read in Data and Helper Funs
ucb <- read_csv("ADNI_data/UCBERKELEYAV1451_01_14_21.csv")
adni <- read_csv("ADNI_data/ADNIMERGE.csv")
source("R/helper_funs.R")

ucb_vol <- dplyr::select(ucb, RID, VISCODE2, dplyr::ends_with("VOLUME"))
ucb_suvr <- dplyr::select(ucb, RID, VISCODE2, dplyr::ends_with("SUVR"))

ucb_vol <- dplyr::rename(ucb_vol, VISCODE = VISCODE2)
ucb_suvr <- dplyr::rename(ucb_suvr, VISCODE = VISCODE2)

# NA for all observations
n <- c(12, 53, 88, 104)

master_ucb_vol <- na.omit(ucb_vol[,-(n)])
master_ucb_suvr <- na.omit(ucb_suvr[, -(n)])

# Get DX Conversions and Merge
#adni <- adniConvert(adni)
#master_ucb_vol <- merge(ucb_vol, adni, by = c("RID", "VISCODE"), all = FALSE)
#master_ucb_suvr <- merge(ucb_suvr, adni, by = c("RID", "VISCODE"), all = FALSE)

readr::write_csv(master_ucb_vol, "processed_data/master_pet_volume.csv")
readr::write_csv(master_ucb_suvr, "processed_data/master_pet_suvr.csv")

# Final Data Dimensions: 1108 x 127
