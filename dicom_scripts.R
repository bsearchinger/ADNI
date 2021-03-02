# Scripts for Reading and Processing DICOM Data

# Install radtools
#source("https://neuroconductor.org/neurocLite.R")
#neuro_install('radtools', release = "stable", release_repo = "github")

library(readr)
library(dplyr)
library(freesurferformats)
library(RNifti)

nii_path <- "~/Dropbox/ADNI/ADNI 4/002_S_0295/FreeSurfer_Cross-Sectional_Processing_brainmask/2006-04-18_08_20_30.0/S13408/ADNI_002_S_0295_VOLUME_FreeSurfer_Cross-Sectional_Processing_Br_20100310173858891.brainmask.nii"
aseg_path <- "~/Dropbox/ADNI/ADNI_aseg/002_S_0295/FreeSurfer_Cross-Sectional_Processing_aparc+aseg/2006-04-18_08_20_30.0/S13408/mri/aparc+aseg.mgz"
mgz_path <- "~/Dropbox/ADNI/FSimgs/002_S_0295/FreeSurfer_Cross-Sectional_Processing_brainmask/2006-04-18_08_20_30.0/S13408/mri/brainmask.mgz"
brain <- read.fs.mgh(mgz_path)

brain_w_head <- read.fs.mgh(mgz_path, with_header = TRUE)


brainmask_nii <- readNifti(nii_path)
brain_aseg <- read.fs.mgh(aseg_path, with_header = TRUE)
brain_aseg$header

h <- niftiHeader(brainmask_nii)
xform(brainmask_nii)
rotation(brainmask_nii)
orientation(brainmask_nii)
ADNI_FS <- read_csv("ADNI_Search_FreeSurfer.csv")

length(unique(ADNI_FS$`Subject ID`))

ADNI_FS_Sum <- ADNI_FS %>%
  group_by(`Research Group`, Visit) %>%
  count()

path <- "AIBL/10/summed.img__RSRCH_RAMLA3D-SUV/2006-10-17_13_53_08.0/I153055/"
dicom_data <- read_dicom(path)
dicom_metadata_matrix <- dicom_header_as_matrix(dicom_data)

MRI <- read_csv("~/Downloads/idaSearch_2_21_2021.csv")
colnames(MRI)
head(MRI)

AD <- MRI %>%
  filter(`Research Group` == "AD")

length(unique(AD$`Subject ID`))
length(unique(MRI$`Subject ID`))

scan_corr_f <- scan_corr[c(1,9)]

filter_df <- comp_1yr %>%
  filter(Sequence %in% scan_corr_f)


p <- "~/Dropbox/ADNI/ADNI/002_S_0295/MPR__GradWarp__B1_Correction__N3__Scaled/2006-04-18_08_20_30.0/S13408/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070319113623975_S13408_I45108.nii"
p2 <- "~/Dropbox/ADNI/ADNI/002_S_0413/MPR__GradWarp__B1_Correction__N3__Scaled/2006-05-02_12_31_52.0/S13893/ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070319115331858_S13893_I45117.nii"

pns <- "~/Dropbox/ADNI/ADNI_NS/002_S_0295/MPR__GradWarp__B1_Correction__N3/2006-04-18_08_20_30.0/S13408/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3_Br_20070319113435616_S13408_I45107.nii"


image <- readNifti(p)
view(image, radiological = TRUE)
xform(image)
rotation(image)
orientation(image)
orientation(image) <- "RAS"
view(image)
dim(image)
pixdim(image)
pixunits(image)
256 * 256
img_mat <- t(as.matrix(image))
dim(img_mat)
nrow(img_mat)

img_small <- image[25:130, 20:90, 80:215]
img_small[1:106, 1, 1]
img_small_mat <- t(as.matrix(img_small))
img_small[,1:3 ,1] == img_small_mat[1,1:316]

long_try <- lapply(seq(1, 106), function(d1){
  layer1 <- lapply(seq(1, 136), function(d2){
    return(img_small[d1, , d2])
  })
  
  layer1 <- dplyr::bind_rows(layer1)
  return(layer1)
  
})

layer1 <- lapply(seq(1, 136), function(d2){
  df <- img_small[ , , d2]
  return(as.data.frame.array(df))
  
})
?as.data.frame
str(layer1[[1]])
layer2 <- data.table::rbindlist(layer1)

layer3 <- prcomp(layer2, center = TRUE, scale. = TRUE)
screeplot(layer3, npcs = 25, type = "lines")
v <- layer3$sdev^2
c <- cumsum(v)

library(FactoMineR)
layer3a <- PCA(layer2, ncp = 15, graph = TRUE)
layer3a$eig
plot.new()
plot(layer3a$var$coord[,2], layer3a$var$coord[,4])
layer3a$var$coord

n1 <- read_nifti1(p)
n2 <- read_nifti1(p2)

ns <- read_nifti1(pns)

img_dimensions(ns)
num_slices(ns)

header_fields(n1)
header_value(n1, "dim_")
nifti1_num_dim(n1)

ns_mat <- img_data_to_mat(ns)

slice80 <- ns_mat[,, 80]

slice80melt <- as.data.frame.table(slice80)
slice80melt$Var2 <- as.integer(slice80melt$Var2)
library(ggplot2)
ggplot(slice80melt, aes(x = Var1, y = Var2, fill = Freq)) +
  geom_raster()

nvals <- nifti1_header_values(n1)
n_image <- img_data_to_mat(n1)
view_slice(ns, 80)
segments(x0 = 100, x1 = 110, y0 = 100, y1 = 100, col = "red")
segments(x0 = 100, x1 = 110, y0 = 110, y1 = 110, col = "red")
segments(x0 = 100, x1 = 100, y0 = 100, y1 = 110, col = "red")
segments(x0 = 110, x1 = 110, y0 = 100, y1 = 110, col = "red")


# ROIs
# Hippocampus
# Some cortex

# Suggestions
# Likes the timeline
# 




