import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from data_handler import generate_subsequences
from matplotlib import pyplot as plt


class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim_rnn, n_layers=1):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_dim_rnn, n_layers)
        self.fc = nn.Linear(hidden_dim_rnn, input_size)


    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)        
        return output
    
    
    
    


sequence_length = 4
hidden_dim_rnn = 32
input_size = dummy_dimenson = 1


hidden_0 = torch.zeros(1, sequence_length, hidden_dim_rnn)
model = RNN(input_size, hidden_dim_rnn)






# generate evenly spaced, test data pts
time_steps = np.linspace(0, np.pi, sequence_length)
data = np.sin(time_steps)

data_T = torch.from_numpy(data).view(1,sequence_length,1).float()




criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

n_epochs = 50
print_every = 2

for batch_i, step in enumerate(range(n_epochs)):
    optimizer.zero_grad()
    hidden_0 = torch.zeros(1, sequence_length, hidden_dim_rnn)
        
    time_steps = np.linspace(step * np.pi, (step + 1) * np.pi, sequence_length + 1)
    data = np.sin(time_steps)
    
    x = data[:-1]
    y = data[1:]
    
    X_T = torch.from_numpy(x).view(1,sequence_length,1).float()
    y_T = torch.from_numpy(y).view(1,sequence_length,1).float()

    
    y_hat = model(X_T, hidden_0)
    
    loss = criterion(y_hat, y_T)
    
    loss.backward()
    optimizer.step()
    
    
    if batch_i % print_every == 0:
        print('Loss: ', loss.item())
    
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(time_steps[1:], y, 'r.', label='y_input')  # input
        ax.plot(time_steps[:-1], y_hat.data.numpy().flatten(), 'b.', label='y_predicted')  # predictions
        ax.legend(loc='best')
        plt.show()
    

    #plt.plot(time_steps, data)
    #plt.show()


























