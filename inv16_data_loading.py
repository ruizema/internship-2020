import numpy as np
import pandas as pd
import pickle
from transforms import transform_log10, norm_1
# Loading kmer table
with open('data/km_table_inv16.py', 'rb') as f:
    kmer_table = pickle.load(f)
# Loading patient table
patient_table = pd.read_csv('data/leucegene.csv', index_col=0, skipfooter=1, engine='python')
patients = patient_table.index.values
# Extracting inv(16) label
inv_16 = [("inv(16)" in text) for text in patient_table['Cytogenetic group']]
inv_16 = [(1 if i else 0) for i in inv_16]
km_flog10 = transform_log10(kmer_table)
km_normed_1 = norm_1(km_flog10)