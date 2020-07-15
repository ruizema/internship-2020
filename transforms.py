import numpy as np
# Log-transform
def transform_log10(x):
    return np.log10(x+1)
# Normalisation
def norm_1(tb):
    '''
    z (standard normal distribution)
    '''
    return (tb - np.array(np.mean(tb, axis=1)).reshape(-1,1))