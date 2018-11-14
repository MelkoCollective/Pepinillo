import numpy as np
from pepinillo.operators import Pauli, Pauli4

# batch, nqubits
samples = np.random.randint(0, 4, (40, 50))
povm = Pauli4()
povm.rho(samples).measure(Pauli.Z, Pauli.X).on(4, 6)
