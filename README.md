# GALLOP-Python

Python implementation of GALLOP algorithm for longitudinal GWAS analysis

## Installation

Install dependencies in requirements.txt with pip. If datatable is not available default to Pandas.

## Basic Usage
```sh

./gallop.py --rawfile $path_to_rawfile \
--pheno $path_to_pheno \
--covar $path_to_covariates \
--time-name study_days \
--pheno-name y \
--covar-name age_at_baseline SEX PC1 PC2 PC3 \
--out $path_to_output

```
