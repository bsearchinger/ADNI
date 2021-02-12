# Scripts for Reading and Processing DICOM Data

# Install radtools
#source("https://neuroconductor.org/neurocLite.R")
#neuro_install('radtools', release = "stable", release_repo = "github")
library(radtools)

path <- "AIBL/10/summed.img__RSRCH_RAMLA3D-SUV/2006-10-17_13_53_08.0/I153055/"
dicom_data <- read_dicom(path)
dicom_metadata_matrix <- dicom_header_as_matrix(dicom_data)





