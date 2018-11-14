import numpy as np
import torch
from torch import optim
from pepinillo.dataloader import POVMData
from pepinillo.operators import Pauli4, Pauli
from rnn import RNN
import _pickle as pkl

device = torch.device("cuda:0" if torch.cuda.device_count() != 0 else "cpu")

n_qubits = 50
n_outcomes =4
batchSize =40
hidden_size = 100
num_gru_layers = 2
num_epochs = 2
log_interval = 50
lr = 0.0001

povm = Pauli4()
dataset = POVMData('../notebooks/data/TFIM_training_data.npz', povm)
dataset = POVMData(dataset.filename, dataset.povm_set, data=dataset.data[:50000])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, num_workers=1)
model = RNN(hidden_size, n_outcomes, n_qubits, num_gru_layers=num_gru_layers, loss_by_step=True, batchSize=batchSize).to(device).double()
optimizer = optim.Adam(model.parameters(), lr=lr)


train_losses = []
KLs = []
fidelities = []
for epoch in range(1, num_epochs + 1):
    #train(epoch, train_losses)
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device).permute(1,0,2)#.reshape(n_qubits, batchSize, n_outcomes)

        #if batch_idx == 1:
         #   print(data[:,0])
        optimizer.zero_grad()
        outputs, hidden, outcome_probs, loss, inputX = model(data)
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data[1]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data[1]) ))
    avg_batch_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_batch_loss)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, avg_batch_loss))


samples = model.sample()
samples = samples.permute([1, 0, 2])
idsamples = np.zeros(batchSize, n_qubits, dtype=np.long)

for i in range(batchSize):
    idsamples[i, :] = np.argmax(samples[i, :, :], axis=1)

measure_X = [povm.rho(idsamples).measure(Pauli.X).on(i) for i in range(n_qubits)]
