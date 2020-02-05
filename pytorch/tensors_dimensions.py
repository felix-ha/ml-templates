import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


number_features = 2
number_classes = 2
number_observations = 100

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(number_features, 10)
        self.fc2 = nn.Linear(10, number_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# Traninig set
x_numpy = np.random.randn(number_observations,number_features)
y_numpy = np.random.randint(1, number_classes, number_observations)

print("Traning set")
print("X", x_numpy)
print("y", y_numpy)

net = Net()
print(net)


print("Sample output")


# Convert numpy array to pytorch tensor
x_sample = x_numpy[0:3,:]
x_sample_tensor = torch.tensor(x_sample).float()
x_sample_tensor = x_sample_tensor.view(-1, number_features)

print("input")
print(x_sample_tensor)

# unsqueeze adds an extra dimension - not used here
#x_sample_tensor = torch.tensor(x_sample).unsqueeze(0).float()
#print(x_sample_tensor)


output = net.forward(x_sample_tensor)

print("output")
print(output)
