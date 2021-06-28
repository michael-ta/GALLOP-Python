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

### Declaring additional model parameters

By default, the model that is fit includes all covariates defined with the `--covar-name` argument. This 
behavior can be overidden by using the `--model` argument to define the exact model to fit. This can be 
used to include higher power or non-linear terms. The genotype will be appended to this into this model.

```sh
--model "y ~ SEX + PC1 + PC2 + PC3 + age_at_baseline + np.power(age_at_baseline, 2) + time"
```

### Fitting multiple phenotypes

If a dataframe with multiple phenotypes is passed into `--pheno` then the `--pheno-name-file` argument
can be used to specify a file with each phenotype to run GALLOP on. Each phenotype in the file must be
on a separate line and appear as is in the column header of the dataframe. One output file for each 
phenotype will be created using the `--out` argument as the base filename and the phenotype as the suffix.

```sh
--pheno $path_to_multi_pheno_df \
--pheno-name-file $path_to_phenotypes
``` 

