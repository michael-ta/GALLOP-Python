#!/usr/bin/env python3
# This script implements the GALLOP algorithm for longitudinal GWAS analysis
# Additional details can be found in the original publication
# https://www.nature.com/articles/s41598-018-24578-7

'''
import datatable
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
'''

import argparse
import sys
import os

def is_plink_export_raw(ds):
  """ Check to see if dataframe comes from PLINK export command, if so we want to drop these columns
      TODO:
        we can either drop columns that are true or drop only if we assume PLINK .raw format
  """
  return ds.columns[:6] == ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']


def preprocess(dp, dc, ds, covariates=None, 
               standardize=True,  keep=None, time_name=None,
               pheno_name=None):
  id_in_study = set.intersection(*[set(x['IID'].tolist()) for x in [dp, dc, ds]])
  sys.stdout.write("Found {} subjects\n".format(len(id_in_study)))
  if keep is not None:
    dtmp = pd.read_csv(keep, engine='c', sep='\t', usecols=['IID'])
    id_in_study = id_in_study.intersection(set(dtmp['IID'].tolist()))

    sys.stdout.write("IID list provided; keeping {} subjects\n".format(len(id_in_study)))
  
  ds = ds[ds.IID.isin(id_in_study)].sort_values(by=['IID'])
  if is_plink_export_raw(ds):
    sys.stdout.write("Detected PLINK raw headers in predictor file; dropping excess columns")
    ds = ds.drop(['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'], axis=1)
  
  df = pd.merge(dp, dc, left_on='IID', right_on='IID', how='inner')
  df = df[df.IID.isin(id_in_study)]

  if time_name is not None:
    df = df[df.IID.isin(id_in_study)].sort_values(by=['IID', time_name])
  else:
    df = df[df.IID.isin(id_in_study)].sort_values(by=['IID'])

  df['time'] = np.subtract(
                np.divide(df.study_days, 365.25),
                np.mean(np.divide(df.study_days.dropna(), 365.25)))

  # scipy scale uses a biased estimator of variance with df=0
  # use an unbiased estimator of variance (n-1)
  data = pd.DataFrame(
          np.divide(scale(df[covariates], with_std=False), 
                          np.matrix(np.sqrt(np.var(df[covariates], ddof=1)))),
          index=df.index, columns=df[covariates].columns)
  
  data['time'] = df.time
  if pheno_name is not None:
    data['y'] = df[pheno_name]
  else:
    data['y'] = df.y
  data['id'] = pd.factorize(df.IID)[0] + 1 # add 1 since R is 1 based indexing

  return data, ds

def fit_lme(formula, data):
  md = smf.mixedlm(formula, data, groups=data["id"], re_formula='~time')
  mdf = md.fit(method=['Powell'], maxiter=1000) # optimizer used by R bobyqa
  return mdf

def main():
  parser = argparse.ArgumentParser(description="""
This script implements the Genome-wide Analysis of Large-scale Longitudinal 
Outcomes using Penalization - GALLOP algorithm

Accepts input files in tab-delimited format. For longitudinal phenotypes, a
column indicating the time point should be specified with the --time-name 
parameter.

For more details, refer to the original publication which can be found at 
https://www.nature.com/articles/s41598-018-24578-7
""",
                                   formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--gallop', help='Fit using GALLOP algorithm; DEFAULT',
                      action='store_true', default=True)
  parser.add_argument('--lmer', help='Fit using LME',
                      action='store_true', default=False)

  parser.add_argument('--preds', help='Predictor file', required=True)
  parser.add_argument('--pheno', help='Phenotype file', required=True)
  parser.add_argument('--covar', help='Covariate file', required=True)

  parser.add_argument('--pheno-name', help='Phenotype or response the data is fit to, DEFAULT=y',
                      default='y')
  parser.add_argument('--covar-name', help='Covariates to include in the model \
                                            ex: "SEX PC1 PC2"',
                      required=True, nargs='+')
  parser.add_argument('--time-name', help='Column name of time variable in phenotype; required for \
                                           GALLOP analysis')
  parser.add_argument('--covar-variance-standardize', 
                      help='Standardize quantitative covariates to follow ~N(0,1)',
                      default=True)
  parser.add_argument('--keep', help='File with IID to keep for analysis')
  parser.add_argument('--out', help='Path to output file')

  args = parser.parse_args()

  dp = pd.read_csv(args.pheno, engine='c', sep='\t')
  dc = pd.read_csv(args.covar, engine='c', sep='\t')
  ds = datatable.fread(args.preds, sep='\t')
  ds = ds.to_pandas()

  # if input is PLINK export format -> we can drop the redundant columns
  # TODO
  #   preprocess data
  #   run either LME or GALLOP algorithm
  

if __name__ == '__main__':
  main()
