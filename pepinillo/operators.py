import numpy as np
from .mps import MPS

__all__ = [
    'Pauli',
    'SICPOVMBase',
    'Pauli4',
    'Tetra',
]

class Pauli:
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1.j], [1.j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


class SICPOVMBase(object):
    """SICPOVMBase

    Base class for SIC-POVM operators.
    """

    # NOTE: this is just a private member stores the rank-3 tensor
    _data = None

    def __getitem__(self, *args):
        return self._data.__getitem__(*args)

    def __repr__(self):
        return self.__title__() + ':\n' + self._data.__repr__()

    def __len__(self):
        return self._data.shape[0]

    def overlap_matrix(self):
        # get the size of this SIC-POVM set
        K = self._data.shape[0]
        T = np.zeros((K, K), dtype=np.complex128)

        # FIXME: use non-loop implementation
        for i in range(K):
            for j in range(K):
                T[i, j] = np.trace(np.matmul(self._data[i, :, :], self._data[j, :, :]))

        return T

    # TODO: expand this to arbitrary observable
    # NOTE: this is just for single qubit observable
    def contract_observable(self, observable):
        invT = np.linalg.inv(self.overlap_matrix())
        # 1. contract M with observable
        out = np.einsum('ijk,jl', self._data, observable)
        # 2. calculate trace
        out = np.einsum('ikk', out)
        # 3. contract with inv(T)
        out = np.matmul(invT, out)
        return out

    def contract_identity(self):
        invT = np.linalg.inv(self.overlap_matrix())
        # 1. trace M since no need to contract identity
        out = np.einsum('ikk', self._data)
        # 2. contract with inv(T)
        out = np.matmul(invT, out)
        return out

    def measure(self, observable, samples):
        return MeasureObservable(self, observable, samples)

    def rho(self, samples):
        return DensityMatrix(self, samples)


class DensityMatrix(object):

    def __init__(self, povm : SICPOVMBase, samples):
        self.povm = povm
        self.samples = samples

    def measure(self, *observables):
        return MeasureObservable(self.povm, observables, self.samples)

    def trace(self, state):
        if isinstance(state, MPS):
            self.trace_mps(state)
        else:
            raise NotImplementedError

    def trace_mps(self, state):
        invT = np.linalg.inv(self.overlap_matrix())
        return        

    def contract(self, state):
        # TODO: contraction with tensor network states
        raise NotImplementedError


class MeasureObservable(object):

    def __init__(self, povm : SICPOVMBase, observables : tuple, samples):
        self.povm = povm
        self.observables = observables
        self.samples = samples

    def on(self, *sites):
        U_O = dict()
        for site, observable in zip(sites, self.observables):
            U_O[site] = self.povm.contract_observable(observable)

        U_I = self.povm.contract_identity()

        def contract_local(sample):
            out = 1.0
            for i, k in enumerate(sample):
                if i in U_O.keys():
                    out = out * U_O[i][k]
                else:
                    out = out * U_I[k]
            return out

        # P_{i_1, i_2, ..., i_N} U_{i_1}^1 U_{i_2}^2 ... U_{i_N}^N
        return np.sum(contract_local(sample) for sample in self.samples) / self.samples.shape[0]

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

    def __title__(self):
        return 'Pauli-4 POVM meansurement'


class Tetra(SICPOVMBase):

    def __init__(self):
        super(Tetra, self).__init__()
        self._data = np.zeros((4, 2, 2), dtype=np.complex128)
        self._data[0, :, :] = 1.0 / 4.0 * (np.eye(2) + Pauli.Z)
        self._data[1, :, :] = 1.0 / 4.0 * (np.eye(2) + 2.0 * np.sqrt(2.0) / 3.0 * Pauli.X - 1.0 / 3.0 * Pauli.Z)
        self._data[2, :, :] = 1.0 / 4.0 * (np.eye(2) - np.sqrt(2.0) / 3.0 * Pauli.X + np.sqrt(2.0 / 3.0) * Pauli.Y - 1.0 / 3.0 * Pauli.Z)
        self._data[3, :, :] = 1.0 / 4.0 * (np.eye(2) - np.sqrt(2.0) / 3.0 * Pauli.X - np.sqrt(2.0 / 3.0) * Pauli.Y - 1.0 / 3.0 * Pauli.Z)

    def __title__(self):
        return 'tetrahedral POVM measurement'
