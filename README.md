# Filtering and Classifying ADNI Participants

## Contributors:

-   [Brian Collica](https://github.com/bcollica)
-   James Koo
-   Ryan Roggenkemper
-   Ben Searchinger

## TLDR

This repository contains R and Python code for reproducing our analysis of data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The model is a two-stage process whereby participants are filtered based their estimated risk for being amyloid-beta positive, and then subsequently classified as either high or low risk for Alzheimer's Disease.

## Data Acquisition

All data required are available through [ADNI](https://adni.loni.usc.edu/) and the Image and Data Archive provided by the [Laboratory of Neuro Imaging](https://www.loni.usc.edu/) at the University of Southern California.

## Repository Structure

The repository has five main directories: `ADNI_data`, `processed_data`, `R`, `nitorch`, and `CNN`. The following raw data files must be downloaded from ADNI and put into `ADNI_data`:

-   `ADNIMERGE.csv`: a collection of key metrics and observations for all ADNI participants
-   `UPENNBIOMK_MASTER.csv`: cerebrospinal fluid analysis (University of Pennsylvania).
-   `UCBERKELEYAV45_01_14_21.csv`: AV45 PET analysis (University of California, Berkeley).
-   `UCBERKELEYFDG_05_28_20.csv`: FDG PET analysis (University of California, Berkeley).
-   `UCBERKELEYAV1451_01_14_21.csv`: tau PET analysis (University of California, Berkeley).
-   `ADNI_UCD_WMH_09_01_20.csv`: white matter hyperintensities (University of California, Davis).
-   `ADNI_PICSLASHS_05_05_20.csv`: medial temporal lobe measurements (University of Pennsylvania).

## Code Replication

Once the data files have been placed in the `ADNI_data` directory, run the following R scripts with the working directory set at the top level of the repo.

### Amyloid Positivity

The R script, `R/amyloid_pos.R`, will output two `.csv` files to the `processed_data` directory: `adnim.csv` and `amyloid_pos_data.csv`.  The dimensions of the files should be 15171 x 58 and 12330 x 57 respectively.

```
insert code chunk here
```

### MRI and PET Volume 

The R scripts, `R/MRI_volume.R` and `R/PET_volume.R`, will output three `.csv` files to the `processed_data` directory: `master_mri_volume.csv`, `master_pet_volume.csv`, and `master_pet_suvr.csv`.  The dimensions of the files should be 1127 x 47, 1108 x 127, and 1108 x 127 respectively.


