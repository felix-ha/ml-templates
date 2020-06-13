import numpy as np


def generate_subsequences(x, sequence_length):
    """
    Given a time series x, returns a prepared data set for a
    sequence-to-sequence model. 
    
    x = (x_0, x_1, ..., x_N) -->
     
    x_batch = (x_i+1, x_i+2, ..., x_sequence_length)
    y_batch = (x_i+2, x_i+3, ..., x_sequence_length+1)
    
    returns a tuple X, y with dimensions 
    [len(x) - sequence_length, sequence_length]
    
    """

    X, y = [], []
    
    for i in range(len(x)):
        x_start = i
        x_end = i+sequence_length     
       
        y_start = x_start+1
        y_end = x_end+1
        
        if y_end > len(x):
            break
        
        x_batch = x[x_start:x_end] 
        y_batch = x[y_start:y_end]
        
        X.append(x_batch)
        y.append(y_batch)
    
    return np.stack(X), np.stack(y)


