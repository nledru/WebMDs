# LV 1/11/22
# purpose is to clean dx_pr_grps data for 2019
# pull data from NIS_2019_DX_PR_GRPS.ASC into csv, referencing code from clean_nis.R
# retrieved txt file from this link: https://www.hcup-us.ahrq.gov/db/nation/nis/tools/stats/FileSpecifications_NIS_2019_DX_PR_GRPS.TXT
# only pull first 23 variables which include comorbidities present on admission
# 2018 data does not contain comorbidities, so will not use
# files needed to run script: NIS_2019_DX_PR_GRPS.ASC, nis_ascii_reader.xlsx, dx_data.txt
# CTRL + F "insert correct pathname" to make appropriate changes

library(tidyverse)
library(LaF)
library(data.table)
library(fs)
library(janitor)
library(readr)
library(openxlsx)

### use file as script
args = commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("specify file-containing directory.n", call = FALSE)
} else {
  dir <- args[1]
}

##### add txt file to xlsx sheet
# import dx 2019 txt file
dx_data <- read_table2(paste0(dir, "/dx_data.txt"),
                       col_names = FALSE)
# remove final columns of strings
dx_data <- dx_data[,1:8]
# rename columns
names(dx_data) <- c("database_name","dc_yr_of_data","file_name",
                    "element_number","var","start_column_ascii",
                    "end_column_ascii","type")
# change var and type to lowercase
dx_data$var <- tolower(dx_data$var)
dx_data$type <- tolower(dx_data$type)
# want 3 columns: width, type, and var
# width = end_column_ascii - start_column_ascii + 1
dx_data$width <- dx_data$end_column_ascii - dx_data$start_column_ascii + 1
# want to change type to correct formats in R; num changes to integer, char changes to string
unique(dx_data$type)
dx_data$type[dx_data$type=="num"] <- "integer"
dx_data$type[dx_data$type=="char"] <- "string"
dx_data <- dx_data %>% select(width, type, var)
dx_data <- dx_data[1:23,]

# create new excel file that contains additional sheet with dx_2019 width, type, and var
#write.csv(dx_data,"/Users/lukevest/Desktop/datathon/dx_ascii_reader.csv") # not necessary, but can manually add csv as a sheet in xlsx file
xlsx_file <- loadWorkbook(paste0(dir, "/nis_ascii_reader.xlsx")) # insert correct pathname
addWorksheet(xlsx_file, "dx_pr_grps")
writeData(xlsx_file, sheet = "dx_pr_grps", x = dx_data)
saveWorkbook(xlsx_file, paste0(dir, "/nis_ascii_reader_dx.xlsx")) # insert correct pathname
#######

###### use Lathan cleaning script
ascii_dx_pr_grps <- readxl::read_excel(paste0(dir, "/nis_ascii_reader_dx.xlsx"), # insert correct pathname
                                       sheet = "dx_pr_grps")

#generates list of variables for cleaning function
#cat(sprintf("%s = fcase(%s >= 0, %s, default = NA),",ascii_dx_pr_grps$var,ascii_dx_pr_grps$var,ascii_dx_pr_grps$var),sep="\n")

# Cleaning Functions ----
clean_and_save_dx_pr_grps <- function(file) {
  file_laf <- laf_open_fwf(file,
                           column_types = ascii_dx_pr_grps$type,
                           column_widths = ascii_dx_pr_grps$width,
                           column_names = ascii_dx_pr_grps$var)
  df <- file_laf[,]
  dt <- as.data.table(df)
  dt[, ':='(hosp_nis = fcase(hosp_nis >= 0, hosp_nis, default = NA),
            key_nis = fcase(key_nis >= 0, key_nis, default = NA),
            cmr_version = fcase(cmr_version >= 0, cmr_version, default = NA),
            cmr_aids = fcase(cmr_aids >= 0, cmr_aids, default = NA),
            cmr_alcohol = fcase(cmr_alcohol >= 0, cmr_alcohol, default = NA),
            cmr_arth = fcase(cmr_arth >= 0, cmr_arth, default = NA),
            cmr_cancer_leuk = fcase(cmr_cancer_leuk >= 0, cmr_cancer_leuk, default = NA),
            cmr_cancer_lymph = fcase(cmr_cancer_lymph >= 0, cmr_cancer_lymph, default = NA),
            cmr_cancer_mets = fcase(cmr_cancer_mets >= 0, cmr_cancer_mets, default = NA),
            cmr_cancer_nsitu = fcase(cmr_cancer_nsitu >= 0, cmr_cancer_nsitu, default = NA),
            cmr_cancer_solid = fcase(cmr_cancer_solid >= 0, cmr_cancer_solid, default = NA),
            cmr_dementia = fcase(cmr_dementia >= 0, cmr_dementia, default = NA),
            cmr_depress = fcase(cmr_depress >= 0, cmr_depress, default = NA),
            cmr_diab_cx = fcase(cmr_diab_cx >= 0, cmr_diab_cx, default = NA),
            cmr_diab_uncx = fcase(cmr_diab_uncx >= 0, cmr_diab_uncx, default = NA),
            cmr_drug_abuse = fcase(cmr_drug_abuse >= 0, cmr_drug_abuse, default = NA),
            cmr_htn_cx = fcase(cmr_htn_cx >= 0, cmr_htn_cx, default = NA),
            cmr_htn_uncx = fcase(cmr_htn_uncx >= 0, cmr_htn_uncx, default = NA),
            cmr_lung_chronic = fcase(cmr_lung_chronic >= 0, cmr_lung_chronic, default = NA),
            cmr_obese = fcase(cmr_obese >= 0, cmr_obese, default = NA),
            cmr_perivasc = fcase(cmr_perivasc >= 0, cmr_perivasc, default = NA),
            cmr_thyroid_hypo = fcase(cmr_thyroid_hypo >= 0, cmr_thyroid_hypo, default = NA),
            cmr_thyroid_oth = fcase(cmr_thyroid_oth >= 0, cmr_thyroid_oth, default = NA)
  )]
  write_csv(dt, gsub(".ASC", ".csv", file))
}

# Run ----
clean_and_save_dx_pr_grps(paste0(dir, "/NIS_2019_DX_PR_GRPS.ASC")) # insert correct pathname
#####

# load dx_pr_grps csv, it contains comorbidities
dx_pr_grps_2019 <- read.csv(paste0(dir, "/NIS_2019_DX_PR_GRPS.csv")) # insert correct pathname

