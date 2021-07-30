#!/usr/bin/env python3
# This script implements the GALLOP algorithm for longitudinal GWAS analysis
# Additional details can be found in the original publication
# https://www.nature.com/articles/s41598-018-24578-7
# Install the python packages in the requirements.txt file prior to running

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
from scipy.stats import norm

import argparse
import sys
import os

try:
  import datatable
except ImportError:
  datatable = None


def is_plink_export_raw(ds):
  """ Check to see if dataframe comes from PLINK export command, if so we want to drop these columns
      TODO:
        we can either drop columns that are true or drop only if we assume PLINK .raw format
  """
  return (ds.columns[:6] == ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']).all()


def load_plink_raw(fn):
  """ More efficient loading of the plink export genotype data
      TODO:
        reduce memory usuage by loading chuncks of data
  """
  cdata = [] # clinical data
  gdata = [] # genotype data
  header = None

  # dtypes of plink headers
  dtypes = {'FID': np.dtype(np.str),
            'IID': np.dtype(np.str),
            'PAT': np.dtype(np.int),
            'MAT': np.dtype(np.int),
            'SEX': np.dtype(np.int),
            'PHENOTYPE': np.dtype(np.int)}
  
  with open(fn, 'r') as f:
    for l in iter(f.readline, ''):
      if header is None:
        header = l.strip().split('\t')
        continue
      data = l.strip().split('\t')
      cdata.append(data[:6])
      tmp_arr = map(lambda x: np.nan if x == 'NA' else x, data[6:])
      tmp_arr = np.asarray(list(tmp_arr), dtype=np.float)
      gdata.append(tmp_arr)
  ds = np.vstack(gdata)
  ds = pd.DataFrame(ds, columns=header[6:])
  cds = pd.DataFrame(cdata, columns=header[:6])
  cds.astype(dtype=dtypes)
  return pd.concat([cds, ds], axis=1)


def preprocess(dp, dc, ds, covariates=None, 
               standardize=True,  keep=None, time_name=None,
               pheno_name=None, MAF=None, rawfile=False, impute=None):

  id_in_study = set.intersection(*[set(x['IID'].tolist()) for x in [dp, dc, ds]])
  sys.stdout.write("Found {} subjects\n".format(len(id_in_study)))
  if keep is not None:
    dtmp = pd.read_csv(keep, engine='c', sep='\t', usecols=['IID'])
    id_in_study = id_in_study.intersection(set(dtmp['IID'].tolist()))

    sys.stdout.write("IID list provided; keeping {} subjects\n".format(len(id_in_study)))
  
  ds = ds[ds.IID.isin(id_in_study)].sort_values(by=['IID'])
  if rawfile:
    ds = ds.drop(['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'], axis=1)
  else:
    ds = ds.drop(['IID'], axis=1) # drop ID after filtering
  
  # drop alleles that have no unique values
  ds_drop_idx = ds.apply(lambda x: x.nunique() == 1)
  ds = ds.drop(ds.columns[ds_drop_idx], axis=1)

  # drop alleles that have no unique values
  ds_drop_idx = ds.isna().all()
  ds = ds.drop(ds.columns[ds_drop_idx], axis=1)

  freq = pd.DataFrame(index = ds.columns)
  n = len(id_in_study) * 2
  allele_freq = ds.apply(lambda x: x.sum() / (n - x.isna().sum()))
  
  if MAF is not None:
    # drop variants that have MAF lower than specified
    ds_drop_idx = np.logical_or( allele_freq < MAF,
                                 allele_freq > (1 - MAF) )
    freq['A1_FREQ'] = allele_freq[~ds_drop_idx]
    ds = ds.drop(ds.columns[ds_drop_idx], axis=1)
  else:
    freq['A1_FREQ'] = allele_freq
  
  missing_freq = None
  if impute is not None:
    imp_ds = SimpleImputer(missing_values=np.nan, strategy=impute)
    imp_ds.fit(ds)
    tmp_header = ds.columns.tolist()
    missing_freq = ds.isna().sum() / len(id_in_study)
    freq['MISS_FREQ'] = missing_freq

    ds = imp_ds.transform(ds)
    ds = pd.DataFrame(ds, columns=tmp_header)
  
  freq = freq.dropna()
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
    data = pd.concat([data, df[pheno_name]], axis=1)
  else:
    data['y'] = df.y
  data['id'] = pd.factorize(df.IID)[0] + 1 # add 1 since R is 1 based indexing
  data.reset_index(drop=True, inplace=True)

  return data, ds, freq

def fit_lme(formula, data):
  md = smf.mixedlm(formula, data, groups=data["id"], re_formula='~time')
  mdf = md.fit(method=['Powell'], maxiter=1000) # optimizer used by R bobyqa
  return mdf

def get_params(mdf):
  ''' given a statsmodel object get the params corresponding to the mean + slope effects
  '''
  tmp_mod.bse['Beta', 'Betai']
  tmp_mod.params['Beta', 'Betai']
  tmp_params.pvalues[['Beta', 'Betai']]


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

  a = pd.DataFrame()
  a['TEST'] = ['ADD_CHANGE'] * ns
  a['BETA'] = -Theta[:,1]
  a['SE'] = SE[:,1]
  a['T_STAT'] = -9
  a['P'] = Pval[:,1]
  a['OBS_CT_REP'] = np.squeeze(np.asarray(np.matmul(TTs[:,0], np.matrix(ds.notna()))))
  a['BETAi'] = -Theta[:,0]
  a['SEi'] = SE[:,0]
  a['Pi'] = Pval[:,0]
  a['COV'] = np.squeeze(COV)

  return a


def do_lme(data, ds, covariates=None):
  s = list(ds.columns)
  ns = len(s)
  beta_incpt = np.empty((ns, 4))
  beta_slope = np.empty((ns, 4))
  
  ids = pd.factorize(data['id'].unique())[0] + 1
  for i in range(0, ns):
    sys.stdout.write(f'Fitting LME for {s[i]}\n')
    tmp_ds = ds[[s[i]]].copy()
    tmp_ds = tmp_ds.rename(columns={s[i]: 'snp'})
    tmp_ds['id'] = ids
    
    tmp_data = data.merge(tmp_ds, on='id', how='inner').copy()
    tmp_data['sxt'] = tmp_data['snp'] * tmp_data['time']
    mod_formula = 'y ~ time + ' + '+'.join(covariates) + ' + snp + sxt' if len(covariates) > 0 else 'snps + sxt'
    mdf = fit_lme(mod_formula, tmp_data)
    beta_incpt[i,:] = np.array([ mdf.params['snp'], mdf.bse['snp'], mdf.tvalues['snp'], mdf.pvalues['snp'] ])
    beta_slope[i,:] = np.array([ mdf.params['sxt'], mdf.bse['sxt'], mdf.tvalues['sxt'], mdf.pvalues['sxt'] ])

  a = pd.DataFrame()
  a['TEST'] = ['ADD_CHANGE'] * ns
  a['BETA'] = -beta_slope[:,0]
  a['SE'] = beta_slope[:,1]
  a['T_STAT'] = beta_slope[:,2]
  a['P'] = beta_slope[:,3]
  a['OBS_CT_REP'] = len(data) # length of the phenotypes includes longitudinal data
  a['BETAi'] = -beta_incpt[:,0]
  a['SEi'] = beta_incpt[:,1]
  a['Pi'] = beta_incpt[:,3]
  return a


def format_gwas_output(ds, res, freq):
  ''' given the genotype dataframe `ds` and the result of the model (gallop or lme) `res` we want to format the 
      output following plink summary statistics

      we're expecting the column names in the dataframe to be chr:pos:ref:ref_alt
  '''
  snp_ids = list(ds.columns)
  data_dict = {'#CHROM': [],
               'ID': [],
               'SNP_ID': [],
               'POS': [],
               'REF': [],
               'ALT': [],
               'A1': []}
  for snp in snp_ids:
    chrom, pos, ref, alt = snp.split(':')
    alt, a1 = alt.split('_')
    ID = ':'.join([chrom, pos, ref, alt])
    data_dict['#CHROM'].append(chrom)
    data_dict['ID'].append(ID)
    data_dict['SNP_ID'].append(snp)
    data_dict['POS'].append(pos)
    data_dict['REF'].append(ref)
    data_dict['ALT'].append(alt)
    data_dict['A1'].append(a1)

  p = pd.DataFrame.from_dict(data_dict)
  p = p.merge(freq.reset_index(), left_on='SNP_ID', right_on='index', how='inner') 
  p['OBS_CT'] = (ds.notna().sum(axis=0)).tolist()
  p = p.drop(columns=['SNP_ID', 'index'])

  result = pd.concat([p, res], axis=1)
  return result
  

def load(fn, hdf_key=None):
  fp, ext = os.path.splitext(fn)
  df = None
  if ext == '.csv':
    df = pd.read_csv(fn, engine='c')
  elif ext == '.h5':
    if hdf_key is None:
      raise KeyError("No HDF key provided")
    df = pd.read_hdf(fn, key=hdf_key)
  elif ext == '.tsv':
    df = pd.read_csv(fn, engine='c', sep='\t')
  else:
    raise Exception("Unknown extension for phenotype")

  df['IID'] = df['IID'].astype(str)
  return df


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
                      action='store_true', default=False)
  parser.add_argument('--lme', help='Fit using LME',
                      action='store_true', default=False)

  parser.add_argument('--rawfile', help='plink RAW file')
  parser.add_argument('--prfile', help='Predictor file')
  parser.add_argument('--hdf-key', help='Key for HDF file')
  parser.add_argument('--pheno', help='Phenotype file', required=True)
  parser.add_argument('--covar', help='Covariate file', required=True)
  parser.add_argument('--model', help='Base model to fit if additional terms needed')

  parser.add_argument('--pheno-name', help='Phenotype or response the data is fit to, DEFAULT=y',
                      default='y')
  parser.add_argument('--pheno-name-file', help='File outlining the phenotypes (one on each line)') 
  parser.add_argument('--covar-name', help='Covariates to include in the model ex: "SEX PC1 PC2"',
                      required=True, nargs='+')
  parser.add_argument('--time-name', help='Column name of time variable in phenotype; required for GALLOP analysis')
  parser.add_argument('--covar-variance-standardize', 
                      help='Standardize quantitative covariates to follow ~N(0,1)',
                      default=True)
  parser.add_argument('--keep', help='File with IID to keep for analysis')
  parser.add_argument('--maf', type=float,
                      help='Filter out variants with minor allele frequency less than maf')
  #parser.add_argument('--pfilter', help='report associations with p-values no greater than threshold')
  #parser.add_argument('--refit', help='refit model with LME if below refit-pval threshold', default=False)
  #parser.add_argument('--refit-pval', help='pvalue threshold for refitting', default=5e-8)
  parser.add_argument('--out', help='Path to output file')

  args = parser.parse_args()

  if not (args.rawfile or args.prfile):
    parser.error('One predictor file must be specified --rawfile for plink RAW and --prfile for other formats')

  rawfile = False
  if args.rawfile:
    rawfile = True

  dp = load(args.pheno, hdf_key=args.hdf_key)
  dc = load(args.covar, hdf_key=args.hdf_key)
  
  if rawfile:
    ds = load_plink_raw(args.rawfile)
  elif datatable is not None:
    ds = datatable.fread(args.rawfile if rawfile else args.prfile, sep='\t')
    ds = ds.to_pandas()
  else:
    ds = pd.read_csv(args.rawfile if rawfile else args.prfile, sep='\t')

  if args.pheno_name_file:
    pheno_name = []
    with open(args.pheno_name_file, 'r') as f:
      pheno_name = f.read().split('\n')
  else:
    pheno_name = [args.pheno_name]

  data, ds, freq = preprocess(dp, dc, ds, args.covar_name, args.covar_variance_standardize, args.keep,
                                      args.time_name, pheno_name, args.maf, rawfile=rawfile, impute='mean')

  if args.gallop or (not args.gallop and not args.lme):
    for pheno in pheno_name:
      XY = data[['id', 'time', pheno] + args.covar_name].copy()
      XY.rename(columns={pheno: 'y'}, inplace=True)
      base_formula = args.model if args.model else f'y ~ {" + ".join(args.covar_name)} + time' 
      sys.stdout.write('Running GALLOP algorithm\n')
      sys.stdout.write(f'Fitting base model: {base_formula}\n') 
      base_mod = fit_lme(base_formula, XY)
      sys.stdout.write(f'Fitting base model with phenotype: {pheno}\n')
      result = do_gallop(base_mod, XY, ds)
      result = format_gwas_output(ds, result, freq)
      out_fn = f'output.{pheno}.gallop'
      if args.out is not None:
        out_fn = f'{args.out}.{pheno}.gallop'
      result.to_csv(out_fn, index=False, sep='\t')

      #if args.refit:
      #  refit_genos = result[(result.P <= args.refit_pval) or (result.Pi <= args.refit_pval)].Transcript.tolist()
      #  sys.stdout.write(f'Refitting {len(refit_genos)} models with p-value < {args.refit_pval}')

      #  for s in refit_genos:
      #    base_formula = args.model if args.model else f'y ~ {"+".join(args.covar_name)} + time'
      #    base_formula += f' + {s} + {s}xt'
      #    tmp_ds = ds[s].reset_index()
      #    tmp_ds['index'] = tmp_ds['index'] + 1
      #    tmp_data = pd.merge(data, tmp_ds, left_on='id', right_on='index', how='inner')
      #    tmp_data[f'{s}xt'] = tmp_data[s] * tmp_data['time']
          
      #    base_mod = fit_lme(base_formiula, tmp_data)
      #    tmp_mod.summary().coef
        
  elif args.lme:
    for pheno in pheno_name:
      data['y'] = data[pheno]
      result = do_lme(data, ds, args.covar_name)
      result = format_gwas_output(ds, result, freq)
      out_fn = f'output.{pheno}.linear_mixed'
      if args.out is not None:
        out_fn = f'{args.out}.{pheno}.linear_mixed'
      result.to_csv(out_fn, index=False, sep='\t')
  # if input is PLINK export format -> we can drop the redundant columns
  # TODO
  #   run either LME or GALLOP algorithm
  

if __name__ == '__main__':
  main()
