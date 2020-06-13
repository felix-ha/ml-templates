import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from data_handler import generate_subsequences
from matplotlib import pyplot as plt


class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim_rnn, n_layers=2):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_dim_rnn, n_layers)
        self.fc = nn.Linear(hidden_dim_rnn, input_size)


    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        
        print("output")
        print(output)
        
        
        print("hidden")
        print(hidden)
        
        
        output_fc = self.fc(output)
        
        print("output fully connected")
        print(output_fc)
        
        return output_fc
    
    
    
    


sequence_length = 10
hidden_dim_rnn = 3
drop_p=0.5
n_layers=1
input_size = dummy_dimenson = 1


hidden_0 = torch.zeros(1, sequence_length, hidden_dim_rnn)
hidden_0 = torch.rand(2, sequence_length, hidden_dim_rnn)
model = RNN(input_size, hidden_dim_rnn)






# generate evenly spaced, test data pts
time_steps = np.linspace(0, np.pi, sequence_length)
data = np.sin(time_steps)




data_T = torch.from_numpy(data).view(1,sequence_length,1).float()



print("input")
print(data_T)


y_hat = model(data_T, hidden_0)

print("")
print("input to model")
print(data_T)
print("output of model")
print(y_hat)




plt.plot(time_steps, data)
plt.show()


























