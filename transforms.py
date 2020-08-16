import numpy as np
# Log-transform
def transform_log10(x):
    return np.log10(x+1)
# Normalisation
def norm_1(tb):
    # Substract by mean of row
    return (tb - np.array(np.mean(tb, axis=1)).reshape(-1,1))/(np.array(np.std(tb, axis=1)).reshape(-1,1)+0.0001)
def norm_2(tb):
    # Normalise by column
    return (tb - np.array(np.mean(tb, axis=0)))/np.std(tb, axis=0)
def norm_3(tb):
    # Normalise from entire dataset
    return (tb - np.mean(np.array(tb)))/np.std(np.array(tb))