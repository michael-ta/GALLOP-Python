#!/usr/bin/env python3
# This script implements the GALLOP algorithm for longitudinal GWAS analysis
# Additional details can be found in the original publication
# https://www.nature.com/articles/s41598-018-24578-7

import datatable
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from scipy.stats import norm

import argparse
import sys
import os

def is_plink_export_raw(ds):
  """ Check to see if dataframe comes from PLINK export command, if so we want to drop these columns
      TODO:
        we can either drop columns that are true or drop only if we assume PLINK .raw format
  """
  return (ds.columns[:6] == ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']).all()


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
    sys.stdout.write("Detected PLINK raw headers in predictor file; dropping excess columns\n")
    ds = ds.drop(['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'], axis=1)
  else:
    ds = ds.drop(['IID'], axis=1) # drop ID after filtering
  
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

def do_gallop(mdf, data, ds):
  G = mdf.cov_re
  # the unexplained variance (residuals) 
  # see doc for class _mixedlm_distribution(object) in 
  # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/regression/mixed_linear_model.py
  sig = np.sqrt(mdf.scale)
  P = np.linalg.inv(np.divide(G, np.power(sig,2)))
  # Given we're running GALLOP these columns should always be present
  # so we want to subset out the covariates from the input data
  covariates = data.columns[~data.columns.isin(['time', 'y', 'id'])]
  covariates = list(covariates)

  XY = data[['time'] + covariates + ['y']]
  XY = sm.add_constant(XY) # add column of 1s for the intercept
  TT = XY.iloc[:,:2]
  nxy = len(XY.columns)
  X = XY.iloc[:,:nxy-1]
  nx = len(X.columns)
  y = XY.iloc[:,nxy-1]
  n = max(data.id)
  ncov = len(covariates)
  A21 = np.empty((2*n, 2 + ncov))
  q2  = np.empty((2*n, 1))
  SS = np.empty((2*n, 2))
  uk = 0

  for i in range(1, n+1):
    k =sum(data.id==i) # N of repeated measurement for subject id
    uk = list(range((uk),(uk+k)))
    u2 = (i * 2) + np.array([-2,-1])
    Ti = TT.iloc[uk,:]
    
    Si = np.matmul(np.transpose(np.matrix(Ti)), np.matrix(Ti))
    u, s, vh = np.linalg.svd(Si + P, compute_uv=True)
    Rot = np.transpose(np.multiply(np.sqrt(1 /s), u))
    Q = np.matmul(Rot, np.matrix(np.transpose(Ti)))
    SS[u2] = np.matmul(Rot, Si)
    A21[u2] = np.matmul(Q,np.matrix(X.iloc[uk,]))
    q2[u2] = np.matmul(Q, np.transpose(np.matrix(y[uk])))
    uk = max(uk) + 1

  q1 = np.matmul(np.transpose(X), y)
  A11 = np.matmul(np.transpose(np.matrix(X)), np.matrix(X))
  # Solve the system (20)
  Q = A11 - np.matmul(np.transpose(A21), A21)
  q = np.subtract(np.matrix(q1).transpose(), np.matmul(np.transpose(A21), q2))
  sol = np.linalg.solve(Q, q)
  blups = q2 - np.matmul(A21, sol)
  sys.stdout.write('GALLOP: blup calculated\n')

  ex = np.ones((1, ncov + 2))
  et = np.ones((1, 2))
  XTk = np.kron(et, X) * np.kron(TT, ex)
  TTk = np.kron(et, TT) * np.kron(TT, et)  
  Tyk = np.matrix(np.multiply(np.array([y,]*2).transpose(), TT))
  XTs = np.zeros((n, XTk.shape[1]))
  TTs = np.zeros((n, TTk.shape[1]))
  Tys = np.zeros((n, 2))
  AtS = np.zeros((n, 2 * nx))
  uk=0
  for i in range(0, n):
    k =sum(data.id==i+1) # N of repeated measurement for i
    uk = list(range((uk),(uk+k)))
    if (k == 1):
      XTs[i, ] = np.matrix(XTk[uk,])
      TTs[i, ] = np.matrix(TTk[uk,])
      Tys[i, ] = np.matrix(Tyk[uk,])
    else:
      XTs[i, ] = XTk[uk, ].sum(axis=0)
      TTs[i, ] = TTk[uk, ].sum(axis=0)
      Tys[i, ] = Tyk[uk, ].sum(axis=0)
    u2 = ((i + 1) * 2) + np.array([-2,-1])
    AtS[i, ] = np.matmul(np.transpose(A21[u2,]), SS[u2,]).flatten(order='F')
    uk = max(uk) + 1

  ns = len(ds.columns)
  s = list(ds.columns)
  Theta = np.empty((ns, 2))
  D = np.empty((ns, 2))
  Corr = np.empty((1,ns))

  for i in range(0, ns):
    si = ds[s[i]]
    miss_vec = np.array(si.isna())
    # unused?
    #miss_vec2 = np.repeat(miss_vec, 2)
    
    si = np.array(si[~miss_vec])
    snp2 = np.repeat(si, 2)
    H1 = np.reshape(np.matmul(si.transpose(), XTs[~miss_vec]), (nx,2), order='F')
    H2 = np.multiply(np.array([snp2,]*2).transpose(), SS)

    AtH = np.reshape(np.matmul(si.transpose(), AtS), (nx,2), order='F')
    R = H1 - AtH
    Cfix = np.linalg.solve(Q, R)
    Cran = H2 - np.matmul(A21,Cfix)

    GtG = np.reshape(np.matmul((si**2).transpose(), TTs), (2,2), order='F')
    Gty = np.reshape(np.matmul(si.transpose(), Tys), (2,1), order='F')
    V = GtG - np.matmul(H1.transpose(), Cfix) - np.matmul(H2.transpose(), Cran)
    v = Gty - np.matmul(H1.transpose(), sol) - np.matmul(H2.transpose(), blups)

    Theta[i] = np.linalg.solve(V, v).flatten()
    Vi = np.linalg.inv(V)
    D[i] = np.diag(Vi)
    Corr[0,i] = Vi[0,1]

  SE = np.multiply(np.sqrt(np.power(sig,2)), np.sqrt(np.abs(D)))
  COV = np.multiply(np.power(sig,2), Corr)
  Pval = np.multiply(2, norm.cdf(-np.abs(Theta / SE)))

  # Export Plink like Formatting
  data_dict = {'CHROM': [],
               'POS': [],
               'REF': [],
               'ALT': [],
               'A1': [],
               'ID': []}
  for snp in s:
    chrom, pos, ref, alt = snp.split(':')
    alt, a1 = alt.split('_')
    id = ':'.join([chrom, pos, ref, alt])
    data_dict['CHROM'].append(chrom)
    data_dict['POS'].append(pos)
    data_dict['REF'].append(ref)
    data_dict['ALT'].append(alt)
    data_dict['A1'].append(a1)
    data_dict['ID'].append(id)

  p = pd.DataFrame.from_dict(data_dict)
  p['A1_FREQ'] = (ds.mean(axis=0) / 2).tolist()
  p['OBS_CT'] = (ds.notna().sum(axis=0)).tolist()
  p['OBS_CT_REP'] = np.squeeze(np.asarray(np.matmul(TTs[:,0], np.matrix(ds.notna()))))

  a = pd.DataFrame()
  a['#CHROM'] = list(map(lambda x: x.replace('chr',''), p['CHROM'].tolist()))
  a['POS'] = p['POS']
  a['ID'] = p['ID']
  a['REF'] = p['REF']
  a['ALT'] = p['ALT']
  a['A1'] = p['ALT']
  a['A1_FREQ'] = round(1-p.A1_FREQ,8)
  a['TEST'] = 'ADD_CHANGE'
  a['OBS_CT'] = p.OBS_CT
  a['BETA'] = -Theta[:,1]
  a['SE'] = SE[:,1]
  a['T_STAT'] = -9
  a['P'] = Pval[:,1]
  a['OBS_CT_REP'] = p.OBS_CT_REP
  a['BETAi'] = -Theta[:,0]
  a['SEi'] = SE[:,0]
  a['Pi'] = Pval[:,0]
  a['COV'] = np.squeeze(COV)

  return a


def do_lme(data, ds):
  s = list(ds.columns)
  ns = len(s)
  betas = np.empty((ns, 4))
  for i in range(0, ns):
    data['snp'] = ds[s[i]]
    mod_formula = 'y ~ time + ' + '+'.join(covariates) + ' + snps' if len(covariates) > 0 else 'snps'
    mdf = fit_lme(mod_formula, data)
    betas[i,:] = np.array([ mdf.params['snp'], mdf.bse['snp'], mdf.tvalues['snp'], mdf.pvalues['snp'] ])

  # Export Plink like Formatting
  data_dict = {'CHROM': [],
               'POS': [],
               'REF': [],
               'ALT': [],
               'A1': [],
               'ID': []}
  for snp in s:
    chrom, pos, ref, alt = snp.split(':')
    alt, a1 = alt.split('_')
    id = ':'.join([chrom, pos, ref, alt])
    data_dict['CHROM'].append(chrom)
    data_dict['POS'].append(pos)
    data_dict['REF'].append(ref)
    data_dict['ALT'].append(alt)
    data_dict['A1'].append(a1)
    data_dict['ID'].append(id)

  p = pd.DataFrame.from_dict(data_dict)
  p['A1_FREQ'] = (ds.mean(axis=0) / 2).tolist()
  p['OBS_CT'] = (ds.notna().sum(axis=0)).tolist()

  a = pd.DataFrame()
  a['#CHROM'] = list(map(lambda x: x.replace('chr',''), p['CHROM'].tolist()))
  a['POS'] = p['POS']
  a['ID'] = p['ID']
  a['REF'] = p['REF']
  a['ALT'] = p['ALT']
  a['A1'] = p['ALT']
  a['A1_FREQ'] = round(1-p.A1_FREQ,8)
  a['TEST'] = 'ADD_CHANGE'
  a['OBS_CT'] = p.OBS_CT
  a['BETA'] = -betas[:,0]
  a['SE'] = betas[:,1]
  a['T_STAT'] = betas[:,2]
  a['P'] = betas[:,3]

  return a


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

  parser.add_argument('--rawfile', help='plink RAW file')
  parser.add_argument('--prfile', help='Predictor file')
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
  parser.add_argument('--keep', help='File with IID to keep for analysis',
    )
  parser.add_argument('--out', help='Path to output file')

  args = parser.parse_args()

  if not (args.rawfile or args.prfile):
    parser.error('One predictor file must be specified --rawfile for plink RAW and --prfile for other formats')

  dp = pd.read_csv(args.pheno, engine='c', sep='\t')
  dc = pd.read_csv(args.covar, engine='c', sep='\t')
  ds = datatable.fread(args.rawfile if not args.prfile else args.prfile, sep='\t')
  ds = ds.to_pandas()

  data, ds = preprocess(dp, dc, ds, args.covar_name, args.covar_variance_standardize, args.keep,
                        args.time_name, args.pheno_name)

  if args.gallop:
    base_formula = f'y ~ {" + ".join(args.covar_name)} + time' 
    sys.stdout.write('Running GALLOP algorithm\n')
    sys.stdout.write(f'Fitting base model: {base_formula}\n') 
    base_mod = fit_lme(base_formula, data)
    result = do_gallop(base_mod, data, ds)
    out_fn = 'gallop_output.csv'
    if args.out is not None:
      out_fn = args.out
    result.to_csv(out_fn, index=False)

  elif args.lme:
    pass
  # if input is PLINK export format -> we can drop the redundant columns
  # TODO
  #   run either LME or GALLOP algorithm
  

if __name__ == '__main__':
  main()
