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

holdout <- read_csv("ConvImgs/holdout.csv")
unique(holdout$`Subject ID`)
holdout_pos <- amyloid_pos %>% filter(Holdout == 1)

adnim <- read_csv("ADNI_data/ADNIMERGE.csv")
adnim_h <- adnim %>% filter(PTID %in% unique(holdout$`Subject ID`))
adnim_b <- adnim_h %>% filter(VISCODE=="bl")
mrid <- adnim_b$RID[which(adnim_b$RID %in% unique(holdout_pos$RID) == F)]
adnim_b %>% filter(RID %in% mrid)
