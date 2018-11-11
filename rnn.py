import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import os
import datetime
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import palettable
import seaborn as sns

# I BUILD THE MODEL, WITH FUNCTIONS FOR INITIALIZING THE HIDDEN STATE AND THE INITIAL X
class RNN(nn.Module):
    def __init__(self, hidden_size, num_gru_layers, n_qubits, n_outcomes, loss_by_step, batchSize):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_gru_layers
        self.steps = n_qubits
        self.n_outcomes = n_outcomes
        self.loss_by_step = loss_by_step
        self.batchSize = batchSize
        
        self.rnn = nn.GRU(n_outcomes, hidden_size, num_layers=self.num_layers, dropout=0.01)
        self.outcome = nn.Linear(hidden_size, n_outcomes)
        self.outcome2 = nn.Linear(n_outcomes, n_outcomes)
        self.CE = nn.CrossEntropyLoss()
    
    def initHidden(self):
        
        '''
        must be shape 
        [self.num_layers * num_directions (1 in this case), batchSize, hidden_size]
        '''
        return torch.ones(1*self.num_layers, batchSize, self.hidden_size).to(device).double()
    
    def initX(self):
        '''
        must be shape 
        [(1 step), batchSize, hidden_size]
        '''
        return torch.ones(1, batchSize, self.n_outcomes).to(device).double()
    

    def step(self, input, hidden=None):
        
        output, hidden = self.rnn(input, hidden)
        for_prediction = output.squeeze(0).to(device)
        outcome = F.softmax(self.outcome(for_prediction), dim = 1)
        return output, hidden, outcome

    def forward(self, inputs, hidden=None, force=True, steps=50):
        
        '''
        must be shape
        input: [length_sequence (n_qubits), batchSize, num_features (n_outcomes)]
        hidden: [num_layers * num_directions, batchSize, hidden_size]'''
        
        if force or steps == 0: steps = len(inputs)
        outputs = torch.zeros(steps, batchSize, hidden_size).double().to(device)
        outcome_probs = torch.zeros(steps, batchSize, n_outcomes).double().to(device)
        losses = torch.tensor(0).double().to(device)

        for i in range(steps):
            if hidden is None and i == 0:
                hidden = self.initHidden()
                input = self.initX()
                inputX = input
                targets = torch.argmax(inputs[i],dim=1)
            else:
                input = inputs[i].unsqueeze(0)
                targets = torch.argmax(inputs[i],dim=1)
                
            #evidence 
                
            output, hidden, outcome = self.step(input, hidden)
            if self.loss_by_step == True:
                #loss_i = self.CE(outcome, targets.long())
                #loss_i = self.step_loss(targets, outcome)
                loss_i = self.myCrossEntropyLoss(outcome, targets)
                losses = losses + loss_i
            outputs[i] = output
            outcome_probs[i] = outcome
            
        if loss_by_step == False:
            losses = self.loss_overall(inputs, outcome_probs)
            
        return outputs, hidden, outcome_probs, losses, inputX


    def step_loss(self, real_outcomes, predicted_outcomes, ce = True):
    
        if ce is True:
            loss = F.cross_entropy(predicted_outcomes, real_outcomes.long(), size_average=False)
        else:
            one_hot_Trueoutcomes = torch.zeros((real_outcomes.shape[0], real_outcomes.max()+1))
            one_hot_Trueoutcomes[torch.arange(real_outcomes.shape[0]).long(),real_outcomes.long()] = 1
            loss = F.mse_loss(one_hot_Trueoutcomes.to(device).double(), predicted_outcomes, size_average=False)
        return loss
    
    def myCrossEntropyLoss(self, outputs, labels):
        batch_size = outputs.size()[0]            # batch_size
        outputs = torch.log(outputs)   # compute the log of softmax values
        outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
        return -torch.sum(outputs)/batch_size

    def loss_overall(self, real_outcomes, predicted_outcomes, ce = True):
    
        if ce is True:
           # CE = F.cross_entropy(real_outcomes[1:], predicted_outcomes[:-1], size_average=False)
            real_outcomes_1 = real_outcomes.view(batchSize*self.steps,4)
            target_labels = torch.argmax(real_outcomes_1,dim=1).long()
            predictions = predicted_outcomes.view(batchSize*self.steps,4)
            
            # take cross entropy between the output prediction and the true target label.
            loss = F.cross_entropy(predictions, target_labels, size_average=False)
        else:
            loss = F.mse_loss(real_outcomes, predicted_outcomes, size_average=False)

        return loss
    
    def sample(self):
    
        #initialize hidden as in model
        hidden = self.initHidden()
        inputX = self.initX()
        samples = torch.zeros(self.steps, self.batchSize, self.n_outcomes)

        for i in range(self.steps):

            #initialize for one-hot encoding, qubit_i_samples will be a class chosen for each batch entry based on probabilities
            # batchSize number of samples for ith qubit
            qubit_i_samples = torch.zeros(self.batchSize)
            #eventually the one hot encoded samples for the ith qubit, num_samples = batchSize
            sample_i = torch.zeros(self.batchSize, self.n_outcomes)
            if i == 0:
                output, hidden, outcome = self.step(inputX, hidden)
                qubit_i_probs = outcome.to('cpu').data.numpy()
            elif i > 0:
                self.step(sample_i.unsqueeze(0).to(device).double(), hidden)

            for batch_entry in range(self.batchSize):
                #implemented w numpy because simpler, no random.choice in torch
                qubit_i_samples[batch_entry] = torch.from_numpy(np.random.choice([0,1,2,3], size = 1, p=qubit_i_probs[batch_entry]))
            sample_i[torch.arange(qubit_i_samples.shape[0]).long(), qubit_i_samples.long()] = 1
            #print(sample_i)

            samples[i] = sample_i

        return samples
    

def load_data(filename):
    """load data from filename
    """
    data_array = np.load(filename)['a']
    reshaped_array = np.empty((1000000, 50, 4))
    for i in range(data_array.shape[0]):
        reshaped_array[i] = data_array[i].reshape((50,4))

    tensor_array_train = torch.stack([torch.Tensor(i).double() for i in reshaped_array[:100000]])
    tensor_data_train = torch.utils.data.TensorDataset(tensor_array_train)
    tensor_array_test = torch.stack([torch.Tensor(i).double() for i in reshaped_array[100000:200000]])
    tensor_data_test = torch.utils.data.TensorDataset(tensor_array_test)

    train_loader = torch.utils.data.DataLoader(tensor_data_train, batch_size=batchSize, num_workers=1)
    test_loader = torch.utils.data.DataLoader(tensor_data_test, batch_size = batchSize, num_workers = 1)
    return train_loader, test_loader


M0 = np.array([[1 / 3, 0], [0, 0]], dtype=np.complex128)
M1 = np.array([[0    , 0], [0, 1/3]], dtype=np.complex128)
M2 = np.array([[1 / 6, 1 / 6], [1 / 6, 1/ 6]], dtype=np.complex128)
M3 = np.array([[0.5 , -1/6], [-1/6, 0.5]], dtype=np.complex128)

M = [M0, M1, M2, M3]

sum(M)

def overlap_matrix(M):
    T = np.zeros(len(M), len(M), dtype=np.complex128)
    for i in range(len(M)):
        for j in range(len(M)):
            T[i, j] = np.trace(np.matmul(M[i], M[j]))

    return T


load_data(''data/numpy_POVM_data.npz'')

device = torch.device("cuda:0" if torch.cuda.device_count() != 0 else "cpu")
print(device)

n_qubits = 50
n_outcomes =4
batchSize =40
hidden_size = 100
num_gru_layers = 2
filename = 'train.txt'
num_epochs = 50
log_interval = 100
loss_by_step = True
lr = 0.001

data_array = np.load('data/numpy_POVM_data.npz')['a']
print(data_array.shape)

# TAKE A LOOK AT WHAT YOUR INPUT DATA LOOKS LIKE, I AM PRINTING THE FIRST ROW (FIRST SAMPLE OF 50 QUBITS)
# I THEN RESHAPE THEM SO EVERY 4 VALUES, WHICH REPRESENT ONE QUBITS OUTCOME MEASUREMENT, APPENDED INTO 50 ROWS
# PER SAMPLE. SO NOW THE DATA IS OF THE SHAPE [n_samples x n_qubits x n_outcomes]
#print(data_array[0])

# I LOAD THE DATA INTO A PYTORCH DATALOADER CLASS

model = RNN(hidden_size).to(device).double()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Training
# train_losses = []
# for epoch in range(1, num_epochs + 1):
#     #train(epoch, train_losses)
#     train_loss = 0
#     for batch_idx, (data) in enumerate(train_loader,):
#         data = data[0].to(device).permute(1,0,2)#.reshape(n_qubits, batchSize, n_outcomes)

#         optimizer.zero_grad()
#         outputs, hidden, outcome_probs, loss, inputX = model(data)
#         loss.backward()
        
#         train_loss += loss.item()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data[1]), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.item() / len(data[1]) ))
#     avg_batch_loss = train_loss / len(train_loader.dataset)
#     train_losses.append(avg_batch_loss)

#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, avg_batch_loss))
