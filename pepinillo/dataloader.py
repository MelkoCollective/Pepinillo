import numpy as np
import torch
from torch.utils.data import Dataset
from .operators import SICPOVMBase


class POVMData(Dataset):
    
    def __init__(self, filename : str, povm_set : SICPOVMBase, data=None):
        self.filename = filename
        self.povm_set = povm_set

        if data is None:
            self.data = torch.tensor(self.load_data(filename, len(povm_set)))
        else:
            self.data = data

    @staticmethod
    def load_data(filename, npovm):
        # TODO: replace `a` with a more readable name
        data = np.load(filename)['data']
        # convention
        # 0: number of samples
        # 1: nsites * npovm
        return data.reshape(data.shape[0], data.shape[1] // npovm , npovm)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.size(0)

    def __repr__(self):
        return 'POVM data of ' + self.povm_set.__title__() + ' size: ' + str(len(self))

    # TODO: assertions
    def to(self, format : str):
        if format == 'onehot':
            data = torch.zeros(self.data.size(0), self.data.size(1), len(self.povm_set))
            for i in range(self.data.size(0)):
                data[i, torch.arange(self.data.size(1)), self.data[i, :].long()] = 1

        elif format == 'idset':
            data = torch.zeros(self.data.size(0), self.data.size(1))
            for i in range(self.data.size(0)):
                data[i, :] = torch.argmax(self.data[i, :, :], dim=1)

        else:
            raise Exception("format string expect \'onehot\' or \'idset\'")

        return POVMData(self.filename, self.povm_set, data)
