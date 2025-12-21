import numpy as np

def onehot(value, max_value):
    vec = np.zeros(max_value)
    vec[value] = 1
    return vec