# GALLOP-Python

Python implementation of GALLOP algorithm for longitudinal GWAS analysis

## Installation

Install packages in requirements.txt with pip

## Basic Usuage
```sh

./gallop.py --rawfile $path_to_rawfile \
--pheno $path_to_pheno \
--covar $path_to_covariates \
--time-name study_days \
--pheno-name y \
--covar-name age_at_baseline SEX PC1 PC2 PC3 \
--out $path_to_output

```
