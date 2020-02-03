import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(3,2))


def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr) // batch_size_total

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n + seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


with open('anna.txt', 'r') as f:
    text = f.read()


text = text[12:12+5]
print(text)


chars = tuple(set(text))
number_unique_chars = len(chars)
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encode the text
encoded = np.array([char2int[ch] for ch in text])

print(chars)
print(int2char)
print(char2int)
print(encoded)


# how many time steps/data pts are in one batch of data
seq_length = 4



print()



# generate evenly spaced data pts
time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data.resize((seq_length + 1, 1)) # size becomes (seq_length+1, 1), adds an input_size dimension

data = one_hot_encode(np.array([encoded]), number_unique_chars)


x = data[0,:-1] # all but the last piece of data
y = data[0,1:] # all but the first

print(data)
print(x)
print(y)


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)

        return output, hidden


#########################################################
# test that dimensions are as expected
test_rnn = RNN(input_size=number_unique_chars, output_size=number_unique_chars, hidden_dim=10, n_layers=1)

# generate evenly spaced, test data pts
#time_steps = np.linspace(0, np.pi, seq_length)
#data = np.sin(time_steps)
#data.resize((seq_length, 1))

test_input = torch.Tensor(x).unsqueeze(0) # give it a batch_size of 1 as first dimension
print('Input size: ', test_input.size())


# test out rnn sizes
test_out, test_h = test_rnn(test_input, None)
print('Output size: ', test_out.size())
print('Hidden state size: ', test_h.size())

print(test_out)

##################################################


# decide on hyperparameters
input_size=number_unique_chars
output_size=number_unique_chars
hidden_dim=32
n_layers=1

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)


# MSE loss and Adam optimizer with a learning rate of 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)


# train the RNN
def train(rnn, n_steps, print_every):
    # initialize the hidden state
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):
        # defining the training data
        data = one_hot_encode(np.array([encoded]), number_unique_chars)

        x = data[0, :-1]  # all but the last piece of data
        y = data[0, 1:]



        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        y_tensor = torch.Tensor(y)



        # outputs from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)

        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()



        # display loss and predictions
        if batch_i % print_every == 0:
            print('Loss: ', loss.item())

            print(prediction)
            print(y_tensor)

            print()

            #fig = plt.figure()
            #ax = fig.add_subplot(1,1,1)
            #ax.plot(time_steps[1:], x, 'r.', label='x')  # input
            #ax.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.', label='y')  # predictions
            #ax.legend(loc='best')
            #plt.show()

    return rnn

# train the rnn and monitor results
n_steps = 30
print_every = 10

trained_rnn = train(rnn, n_steps, print_every)



