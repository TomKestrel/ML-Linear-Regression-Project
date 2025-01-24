import numpy as np

def load_data():
    data = np.loadtxt("data/sqft_vs_cost_data.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y
