import numpy as np

class MPS(object):
    """(Open boundary) Matrix Product State
    """

    def __init__(self, *tensors):
        # TODO : assertion on input tensors, should be rank-3 and first dim is 2
        self.tensors = tensors
        self.system_size = len(tensors)

    def left_contraction(self, configs):
        """contract from left hands
        """
        out = self.tensors[0][configs[0]]
        for i in range(1, self.system_size):
            out = np.matmul(out, self.tensors[i][configs[i]])

        return np.trace(out)

    def right_contraction(self, configs):
        """contract from right hands
        """
        out = self.tensors[-1][configs[-1]]
        for i in range(self.system_size - 2, -1, -1):
            out = np.matmul(out, self.tensors[i][configs[i]])

        return np.trace(out)

    def amplitude(self, configs=None):
        if configs is None:
            return self.state()
        else:
            return self.left_contraction(configs)

    def state(self):
        out = self.tensors[0]
        for (i, each) in enumerate(self.tensors[1:]):
            out = np.tensordot(out, each, axes=([2, ], [1, ]))
            out = out.reshape(2**(i + 2), out.shape[-2], out.shape[-1])

        return out.trace(axis1=1, axis2=2)
