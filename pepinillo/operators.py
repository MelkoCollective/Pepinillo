import numpy as np

class SICPOVMBase(object):
    """SICPOVMBase

    Base class for SIC-POVM operators.
    """

    # NOTE: this is just a private member stores the rank-3 tensor
    _data = None

    def __getitem__(self, *args):
        return self._data.__getitem__(*args)

    def __repr__(self):
        raise NotImplementedError

    def overlap_matrix(self):
        # get the size of this SIC-POVM set
        K = self._data.shape[0]
        T = np.zeros((K, K), dtype=np.complex128)

        # FIXME: use non-loop implementation
        for i in range(K):
            for j in range(K):
                T[i, j] = np.trace(np.matmul(M[i, :, :], M[j, :, :]))

        return np.linalg.inv(T)

    # TODO: expand this to arbitrary observable
    # NOTE: this is just for single qubit observable
    def contract_observable(self, observable):
        T = self.overlap_matrix()
        out = np.einsum('ijk,jl', self._data, observable)
        out = np.einsum('ikk', out)
        out = np.matmul(T, out)
        return out

    def measure_observable(self, observable, povm_samples):
        O = self.contract_observable(observable)
        return np.sum(np.sum(O[i] for i in each) for each in samples) / samples.shape[0]


class Pauli4(SICPOVMBase):

    def __init__(self):
        super(Pauli4, self).__init__()
        self._data = np.zeros((4, 2, 2), dtype=np.complex128)
        self._data[0, :, :] = 1.0 / 3.0 * np.array([[1, 0], [0, 0]])
        self._data[1, :, :] = 1.0 / 6.0 * np.array([[1, 1], [1, 1]])
        self._data[2, :, :] = 1.0 / 6.0 * np.array([[1, -1j], [1j, 1]])
        self._data[3, :, :] = 1.0 / 3.0 * (np.array([[0, 0],[0, 1]]) + \
                                        0.5*np.array([[1, -1],[-1, 1]]) + \
                                        0.5*np.array([[1, 1j],[-1j, 1]]))

    def __repr__(self):
        return 'Pauli-4 POVM meansurement:\n' + self._data.__repr__()
