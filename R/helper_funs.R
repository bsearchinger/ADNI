# Helper Functions for Collecting and Parsing ADNI Data


# ADNI Convert ####
# Function To Create Indicator Variables for Diagnosis Conversions
# Used for Merging Into Other Datasets by RID and VISCODE
adniConvert <- function(dat, dx_only = TRUE) {
  dat <- dat %>% 
    dplyr::mutate(
      AD_CONV = ifelse(DX_bl != "AD" & DX == "Dementia", 1, 0),
      CN_AD = ifelse(DX_bl == "CN" & DX == "Dementia", 1, 0),
      CN_MCI = ifelse(DX_bl == "CN" & DX == "MCI", 1, 0),
      EMCI_AD = ifelse(DX_bl == "EMCI" & DX == "Dementia", 1, 0),
      LMCI_AD = ifelse(DX_bl == "LMCI" & DX == "Dementia", 1, 0))
  
  if (dx_only) {
    dat <- dat %>% dplyr::select(
      RID, VISCODE, DX_bl, DX, AD_CONV, CN_AD, CN_MCI, EMCI_AD, LMCI_AD)
  }
  
  return(dat)
}

