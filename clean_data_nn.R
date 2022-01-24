data_dir <- "~/Desktop/hcup/"

library(here)
library(stringr)
library(magrittr)
library(future)
library(parallel)
library(future.apply)
library(tidyverse)

core_2019 <- read_csv(here(data_dir, "NIS_2019_Core.csv"))
severity_2019 <- read_csv(here(data_dir, "NIS_2019_Severity.csv"))
hospital_2019 <- read_csv(here(data_dir, "NIS_2019_Hospital.csv"))
cc_2019 <- read_csv(here(data_dir, "cc2019NIS.csv"))

# extract unique ICD-10 codes from core csv for use as column names
one_hot_encode_mat <- function(core_csv, dx_num = 40, pr_num = 25, trim_by_hierarchy = FALSE, trim_depth = 3) {
    unique_icd10_dxs <- unique(unlist(lapply(str_c("i10_dx", 1:dx_num), function(x) do.call('$', list(core_csv, x)))))
    unique_icd10_prs <- unique(unlist(lapply(str_c("i10_pr", 1:pr_num), function(x) do.call('$', list(core_csv, x)))))
    if ("" %in% unique_icd10_dxs) { unique_icd10_dxs <- unique_icd10_dxs[-which(unique_icd10_dxs == "")] } # depending on whether empty columns are "" or NA, will need to remove "" from list
    if ("" %in% unique_icd10_prs) { unique_icd10_prs <- unique_icd10_prs[-which(unique_icd10_prs == "")] } # same

    if (trim_by_hierarchy) {
        trim_dxs <- unique(str_sub(unique_icd10_dxs, start = 1, end = trim_depth))
        trim_prs <- unique(str_sub(unique_icd10_prs, start = 1, end = trim_depth))
    } else {
        trim_dxs <- unique_icd10_dxs
        trim_prs <- unique_icd10_prs
    }

    # construct sparse mat
    icd10_colnames <- c(str_c("i10_dx", 1:dx_num), str_c("i10_pr", 1:pr_num))
    core_csv_icdcols <- core_csv[, icd10_colnames] %>% t() # couple steps to speed up processing
    plan(multicore, workers = 10, gc = TRUE) # change to multisession/cluster if on rstudio
    options(future.globals.maxSize = 400 * 1024 ^ 2)
    colname_list <- future_apply(core_csv_icdcols, 2, function(x) {
                                                            if (trim_by_hierarchy) {
                                                                x[which(!is.na(x))] %>% as.character() %>% str_sub(start = 1, end = trim_depth)
                                                            } else {
                                                                x[which(!is.na(x))] %>% as.character()        
                                                            }
                                                       }) 
    colname_list_tuple <- future_mapply(list, colname_list, 1:length(colname_list), SIMPLIFY = FALSE) # ugly, but need this to pass row number to sparse matrix construction without using tons of RAM
    # rm(colname_list); gc() # RAM management
    sparse_mat_list <- future_lapply(colname_list_tuple, function(x) {
                                                            data.frame(i = rep(x[[2]], length(x[[1]])), j = x[[1]]) # i is row number, j is column for unique icd-10 code
                                                         })
    sparse_mat_icd <- bind_rows(sparse_mat_list) %>% mutate(x = 1) ## add x column now, as all values are 1
    # rm(sparse_mat_list); gc() # RAM management
    icd_dim_dict <- 1:length(c(trim_dxs, trim_prs))
    names(icd_dim_dict) <- c(trim_dxs, trim_prs)
    sparse_mat_icd$j <- icd_dim_dict[sparse_mat_icd$j] # need to convert ICD strings to numeric for sparse mat conversion

    icd_mat <- Matrix::sparseMatrix(i = sparse_mat_icd$i, j = sparse_mat_icd$j, x = sparse_mat_icd$x, dims = list(dim(core_csv)[1], length(icd_dim_dict))) 
    colnames(icd_mat) <- names(icd_dim_dict)
    icd_mat <- icd_mat[, which(Matrix::colSums(icd_mat) > 10)] # IMPORTANT: per DUA remove cols with <= 10 entries
    plan(sequential)
    return(icd_mat)
}

clean_cores <- function(core_csv, coreyear, dx_num = 40, pr_num = 25) {
    icd10_colnames <- c(str_c("i10_dx", 1:dx_num), str_c("i10_pr", 1:pr_num))
    prday_colnames <- c(str_c("prday", 1:pr_num))
    core_csv <- core_csv %>% select(-c(icd10_colnames, prday_colnames))
    if (coreyear == 2016) {
        core_csv <- core_csv %>% select(-c(i10_ecause1, i10_ecause2, i10_ecause3, i10_ecause4))
    }
    # core_csv <- core_csv[which(is.na(core_csv$age_neonate)), ]
    # core_csv <- core_csv %>% select(-age_neonate)
    # core_csv <- core_csv[which(rowSums(is.na(core_csv)) == 0), ]
    core_csv <- core_csv %>% mutate(year = coreyear)
    return(as.matrix(core_csv, sparse = TRUE))
}

ohe2019 <- one_hot_encode_mat(core_2019)
clean_2019 <- clean_cores(core_2019, 2019)
clean_2019 <- cbind(clean_2019, ohe2019) 

# checkpoint
saveRDS(clean_2019, "ohe_core_2019.RDS")

##### second cleaning attempt
### similar to Jess's removal of CMR data, removing ICD codes with < 10000 instances across 7 million rows (~1547 were present in 1% rows, 10k filter leaves 1214)
allvar_2019 <- readRDS("ohe_core_2019.RDS")
clean2019_allrow  <- allvar_2019[which(is.na(allvar_2019[, "age_neonate"])),]
icd_kept_cols <- which(colSums(clean2019_allrow[, colnames(clean2019_allrow)[63:length(colnames(clean2019_allrow))]]) > 10000)
clean2019_allrow <- clean2019_allrow[, c(colnames(clean2019_allrow)[1:62], names(icd_kept_cols))] # filtering out after removing neonates results in 1218 total columns 
clean2019_allrow <- clean2019_allrow[, which(!(colnames(clean2019_allrow) %in% c("age_neonate", "dqtr", "drg", "drgver", "drg_nopoa", "i10_birth", "i10_delivery",
                                                                                 "i10_serviceline", "mdc", "mdc_nopoa",
                                                                                 c(str_c("prday", 1:25)))))]
clean2019_allrow <- clean2019_allrow[which(rowSums(is.na(clean2019_allrow)) == 0), ]
# dataset is 6286419 before removing any rows with at least one NA
# 5943380 with no NAs - all but ~30k are race; distribution for race stil has 67% white, 15% black, 11% hispanic, etc seems closeish to just filter all NAs for now

clean2019_allrow <- clean2019_allrow[which(rowSums(clean2019_allrow[, 28:length(colnames(clean2019_allrow))]) > 0), ] # getting rid of samples with 0 icd codes in the remaining ones--basically going to try to learn cost of most common icd codes and avoid a fake intercept with wide variance
# after this, 5901142 by 1183

## add other data
ccdf <- as.data.frame(cc_2019)
hospdf <- as.data.frame(hospital_2019)
sevdf <- as.data.frame(severity_2019)
cmrdf <- as.data.frame(read_csv(here(data_dir, "NIS_2019_DX_PR_GRPS.csv")))
rownames(ccdf) <- ccdf$hosp_nis
rownames(hospdf) <- hospdf$hosp_nis
rownames(cmrdf) <- cmrdf$key_nis
if (all.equal(cmrdf$key_nis, sevdf$key_nis)) { 
    rownames(sevdf) <- rownames(cmrdf)
}
hospdf <- hospdf[, which(!(colnames(hospdf) %in% c("year", "hosp_nis")))]
cmrdf <- cmrdf[, 4:23]

kept_keys <- as.character(clean2019_allrow[, "key_nis"])
cmrsm <- as.matrix(cmrdf, sparse = TRUE)
cmrsm_sub <- cmrsm[kept_keys, ]

sevsm <- as.matrix(sevdf, sparse = TRUE)
sevsm_sub <- sevsm[kept_keys, ]

combsm <- cbind(cmrsm_sub, sevsm_sub)

clean2019_allrow <- cbind(ccdf[as.character(clean2019_allrow[, "hosp_nis"]), c("ccr_nis", "wageindex")] %>% as.matrix(sparse = TRUE), clean2019_allrow)
clean2019_allrow <- cbind(hospdf[as.character(clean2019_allrow[, "hosp_nis"]), ] %>% as.matrix(sparse = TRUE), clean2019_allrow)
final2019 <- cbind(combsm, clean2019_allrow)
# saveRDS(final2019, "final2019.RDS") - not enough RAM to do all at once, will do in chunks

final2019_write <- as.matrix(final2019[1:1000000, ])
write.csv(final2019_write, "HR_2019_split/HR_chunk0.csv", row.names = FALSE)
rm(final2019_write); gc()

final2019_write <- as.matrix(final2019[1000001:2000000, ])
write.csv(final2019_write, "HR_2019_split/HR_chunk1.csv", row.names = FALSE)
rm(final2019_write); gc()

final2019_write <- as.matrix(final2019[2000001:3000000, ])
write.csv(final2019_write, "HR_2019_split/HR_chunk2.csv", row.names = FALSE)
rm(final2019_write); gc()

final2019_write <- as.matrix(final2019[3000001:4000000, ])
write.csv(final2019_write, "HR_2019_split/HR_chunk3.csv", row.names = FALSE)
rm(final2019_write); gc()

final2019_write <- as.matrix(final2019[4000001:5000000, ])
write.csv(final2019_write, "HR_2019_split/HR_chunk4.csv", row.names = FALSE)
rm(final2019_write); gc()

final2019_write <- as.matrix(final2019[5000001:dim(final2019)[1], ])
write.csv(final2019_write, "HR_2019_split/HR_chunk5.csv", row.names = FALSE)
rm(final2019_write); gc()

