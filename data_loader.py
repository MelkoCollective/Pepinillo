import numpy as np
import torch
from torch.utils.data import Dataset

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


class POVMData(Dataset):
    def __init__(self, filename, train_len=100000, dtype=torch.float64):
        self.filename = filename

        raw_data = self.load_raw_data(filename)
        self.train_tensors = torch.stack([torch.Tensor(each, dtype=dtype) for each in raw_data[:train_len]])

    @staticmethod
    def load_raw_data(filename):
        # TODO: use a better name
        return np.load(self.filename)['a']

    def __len__(self):
        return self.train_tensors[0].size(0)

    def __getitem__(self):
        return tuple(self.train_tensors[index] for self.self.tensors)

tensor_array_test
