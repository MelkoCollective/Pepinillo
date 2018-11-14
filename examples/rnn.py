import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class RNN(nn.Module):
    def __init__(self, hidden_size, n_outcomes, steps, num_gru_layers=2, loss_by_step=True, batchSize=None, device=None):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_gru_layers
        self.steps = steps
        self.n_outcomes = n_outcomes
        self.loss_by_step = loss_by_step
        self.device = device

        if batchSize is None:
            raise ValueError

        self.batchSize = batchSize

        
        self.rnn = nn.GRU(n_outcomes, hidden_size, num_layers=self.num_layers, dropout=0.01)
        self.outcome = nn.Linear(hidden_size, n_outcomes)
        #self.outcome2 = nn.Linear(hidden_size // 2, n_outcomes)
        self.CE = nn.CrossEntropyLoss()
        
        
    
    def initHidden(self):
        
        '''
        must be shape 
        [self.num_layers * num_directions (1 in this case), batchSize, hidden_size]
        '''
        return torch.zeros(1*self.num_layers, self.batchSize, self.hidden_size).to(self.device).double()
    
    def initX(self):
        '''
        must be shape 
        [(1 step), batchSize, hidden_size]
        '''
        return torch.zeros(1, self.batchSize, self.n_outcomes).to(self.device).double()
    

    def step(self, input, hidden):
        
        output, hidden = self.rnn(input, hidden)
        for_prediction = output.squeeze(0).to(self.device)
        outcome_prob = F.softmax(self.outcome(for_prediction), dim = 1)
        return output, hidden, outcome_prob

    
    def forward(self, inputs, hidden=None,  steps=50):
        
        '''
        must be shape
        input: [length_sequence (n_qubits), batchSize, num_features (n_outcomes)]
        hidden: [num_layers * num_directions, batchSize, hidden_size]'''
        
        if steps == 0: steps = len(inputs)
        outputs = torch.ones(steps, self.batchSize, self.hidden_size).double().to(self.device)
        outcome_probs = torch.ones(steps, self.batchSize, self.n_outcomes).double().to(self.device)
        #losses = torch.tensor(1).double().to(self.device)
        losses = torch.zeros(steps, self.batchSize)

        for i in range(steps):
            if  i == 0:
                hidden = self.initHidden()
                input = self.initX()
                inputX = input
                targets = torch.argmax(inputs[i],dim=1)
                
            else:
                input = inputs[i-1,:,:].unsqueeze(0)
                targets = torch.argmax(inputs[i],dim=1)
                

                
            output, hidden, outcome = self.step(input, hidden)

            if self.loss_by_step == True:
                #loss_i = self.CE(outcome, targets.long())
                #loss_i = self.step_loss(targets, outcome)
                loss_i = self.myCrossEntropyLoss(outcome, targets)
                losses[i,:] = loss_i
            outputs[i] = output
            outcome_probs[i] = outcome
            
        if self.loss_by_step == False:
            losses = self.loss_overall(inputs, outcome_probs)
            
        loss = torch.sum(losses)
        return outputs, hidden, outcome_probs, loss, inputX
       

    def myCrossEntropyLoss(self, outputs, labels):
        batch_size = outputs.size()[0]            # batch_size
        outputs = torch.log(outputs)   # compute the log of softmax values
        outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
        #print(outputs.shape)
        return -torch.sum(outputs)/batch_size
    
    
    def sample(self):
    
        #initialize hidden as in model
        hidden = self.initHidden()
        inputX = self.initX()
        
        # one hot encoded for each qubit
        samples = torch.zeros(self.steps, self.batchSize, self.n_outcomes)
        probs = torch.zeros(self.batchSize)

        #go through all qubits
        for i in range(self.steps):

            #initialize for one-hot encoding, qubit_i_samples will be a class chosen for each batch entry based on probabilities
            # batchSize number of samples for ith qubit
            qubit_i_samples = torch.zeros(self.batchSize)
            if i == 0:
                output, hidden, outcome = self.step(inputX, hidden)
                qubit_i_probs = outcome.to('cpu').data.numpy()
            elif i > 0:
                output, hidden, outcome = self.step(input.unsqueeze(0).to(self.device).double(), hidden)
                qubit_i_probs = outcome.to('cpu').data.numpy()

            for batch_entry in range(self.batchSize):

                qubit_i_samples[batch_entry] = torch.from_numpy(np.random.choice([0,1,2,3], size = 1, p=qubit_i_probs[batch_entry]))

            #make the one-hot samples, use to feed as next input
            samples[i][torch.arange(qubit_i_samples.shape[0]).long(), qubit_i_samples.long()] = 1
            input = samples[i]

        return samples
    

    def step_loss(self, real_outcomes, predicted_outcomes, ce = True):
    
        if ce is True:
            loss = F.cross_entropy(predicted_outcomes, real_outcomes.long(), size_average=False)
        else:
            one_hot_Trueoutcomes = torch.zeros((real_outcomes.shape[0], real_outcomes.max()+1))
            one_hot_Trueoutcomes[torch.arange(real_outcomes.shape[0]).long(),real_outcomes.long()] = 1
            loss = F.mse_loss(one_hot_Trueoutcomes.to(self.device).double(), predicted_outcomes, size_average=False)
        return loss
