import torch

x = torch.zeros(5, 3, dtype=torch.long)
y = torch.rand(5, 3)
print(x)
print(y)
print(x+y)

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# _ mutates a tensor in-place
y.add_(x)
print(y)

#indexing numpy-like
print(x[:, 1])

#reshape with view

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

#one element tensor
x = torch.randn(1)
print(x)
print(x.item())

#Numpy - PyTorch
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# Converting Numpy array to Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# cuda tensors - are not available on this machine
print(torch.cuda.is_available())