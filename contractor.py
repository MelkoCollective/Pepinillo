import numpy as np

def pauli_4():
    M = np.zeros((4, 2, 2), dtype=np.complex128)
    M[0, :, :] = 1.0 / 3.0 * np.array([[1, 0], [0, 0]])
    M[1, :, :] = 1.0 / 6.0 * np.array([[1, 1], [1, 1]])
    M[2, :, :] = 1.0 / 6.0 * np.array([[1, -1j], [1j, 1]])
    M[3, :, :] = 1.0 / 3.0 * (np.array([[0, 0],[0, 1]]) + \
                                    0.5*np.array([[1, -1],[-1, 1]]) + \
                                    0.5*np.array([[1, 1j],[-1j, 1]]))
    return M


def inv_overlap_matrix(M):
    K = M.shape[0]
    T = np.zeros((K, K), dtype=np.complex128)
    
    for i in range(K):
        for j in range(K):
            T[i, j] = np.trace(np.matmul(M[i, :, :], M[j, :, :]))
    
    return np.linalg.inv(T)


def contracted_TM_observable(T, M, op):
    Mop = np.einsum('ijk,jl', M, op)
    Mop = np.einsum('ikk', Mop)
    TMop = np.matmul(T, Mop)
    return TMop

def contract_probability(O, samples):
    return np.sum(np.sum(O[i] for i in each) for each in samples) / samples.shape[0]


samples = np.random.randint(0, 4, size=(10, 4))

M = pauli_4()
T = inv_overlap_matrix(M)
Z = np.array([[1, 0], [0, -1]])
O = contracted_TM_observable(T, M, Z)
contract_probability(O, samples)
