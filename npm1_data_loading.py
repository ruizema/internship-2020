import numpy as np
import pandas as pd
import pickle
from transforms import transform_log10, norm_1
# Loading kmer table
with open('data/km_table_npm1.py', 'rb') as f:
    kmer_table = pickle.load(f)
# Loading patient table
patient_table = pd.read_csv('data/leucegene.csv', index_col=0, skipfooter=1, engine='python')
patients = patient_table.index.values
# Extracting label
npm1 = [(-1 if i == '-' else int(i)) for i in patient_table['NPM1 mutation']]
# Filtering out patients with no data
kmer_table_filtered = kmer_table.copy()
for row in npm1:
    if row == -1:
        kmer_table_filtered = kmer_table_filtered.drop(kmer_table_filtered.index[row])
patient_list_filtered = []
npm1_filtered = []
for i in range(len(npm1)):
    if npm1[i] != -1:
        patient_list_filtered.append(patients[i])
        npm1_filtered.append(npm1[i])
km_flog10 = transform_log10(kmer_table_filtered)
km_normed_1 = norm_1(km_flog10)