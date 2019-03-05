# Lab 12 RNN
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

torch.manual_seed(666)  # reproducibility




idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3],
          [2, 2, 2, 2, 2, 2],
          [0, 1, 0, 1, 0, 1],
          [1, 2, 3, 4, 4, 4],
          [0, 2, 4, 4, 4, 3],
          [1, 4, 0, 4, 4, 3],
          [4, 4, 3, 2, 1, 0],
          [3, 2, 3, 4, 0, 1]
         ]   # hihell

y_data = [[1,0,2,3,3,4],
          [4,4,4,4,4,4],
          [1,0,1,0,1,0],
          [0,2,3,3,3,3],
          [3,4,4,4,4,3],
          [3,2,2,2,4,1],
          [0,2,4,1,4,3],
          [2,1,0,2,2,2]
         ]
          
          # ihello


data_array = np.load('data/TFIM_training_data.npz')['data']
log_probs = np.load('data/TFIM_logprobs.npz')['probs']

NumOfMolecules=len(data_array)

print("Num of molecules", NumOfMolecules)

# [num_samples, seq_length, num_outcomes] # one hot encoding each sequence
# x_OH=np.zeros((NumOfMolecules,len(x_data[0]),len(idx2char)))
# for jj in range(NumOfMolecules):
#     for ii in range(len(x_data[jj])):
#     	x_OH[jj][ii][x_data[jj][ii]]=1


x_OH = np.empty((NumOfMolecules, 50, 4))
for i in range(data_array.shape[0]):
    x_OH[i] = data_array[i].reshape((50,4))

    

print(x_OH[0])


# As we have one batch of samples, we will change them to variables only once
inputs = Variable(torch.Tensor(x_OH))
labels = Variable(torch.LongTensor(y_data))

idx2char = [0,1,2,3]
input_size = len(idx2char)    # one-hot size

batch_size = 50                     # one sentence
sequence_length = 1 #len(y_data)   # |ihello| == 6
num_layers = 3                  # one-layer rnn
hidden_size = len(idx2char)     # output from the LSTM. 5 to directly predict one-hot
                                # TODO: Add fully connected layer
output_size = hidden_size       # Len of the output one-hot

class Model(nn.Module):

   def __init__(self):
       super(Model, self).__init__()
       self.rnn = nn.GRU(input_size=input_size, 
                  hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

   def forward(self, hidden, x):
       # Reshape input in (1, sequence_length, input_size)
       x = x.view(1, sequence_length, input_size)

       # Propagate input through RNN
       # Input: (batch, seq_len, input_size)
       # hidden: (batch, num_layers * num_directions, hidden_size)
       out, hidden = self.rnn(x, hidden)
       out = out.view(-1, input_size)
       return hidden, out

   def init_hidden(self):
       # Initialize hidden and cell states
       # (num_layers * num_directions, 1, hidden_size) for batch_first=True
       return Variable(torch.zeros(num_layers, 1, hidden_size))
   
    
    
    
    # Instantiate RNN model
model = Model()

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1):
    optimizer.zero_grad()

    hidden = model.init_hidden()
   
    BatchIteration=0
    for kk in range(int(len(inputs)/batch_size)): # runs over different batches
        loss = 0
        for jj in range(batch_size):   
            CurrentSMILES=jj+BatchIteration*batch_size                  # runs over SMILES from batch
            for ii in range(len(labels[0])):                # runs over letters from SMILES
            #for input_char, label_char in zip(inputs, labels):
                # print(input.size(), label.size())
                input_char = inputs[CurrentSMILES][ii]
                label_char = labels[CurrentSMILES][ii].reshape(1)
 
                hidden, output = model(hidden, input_char)
                val, idx = output.max(1)
                loss += criterion(output, label_char)
            #print('jj in batch_size: ', jj)     
            #print('batch num: ', kk)
            
        BatchIteration+=1
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(),0.5)
        optimizer.step()
        

    print("Epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))


   
   
   
# TESTING
model.eval()


hidden = model.init_hidden()
for ii in range(len(y_data[0])):
    hidden, prediction = model(hidden, inputs[0][ii])
    #print(prediction)
    _, idx = prediction.max(1)
    print(idx)


