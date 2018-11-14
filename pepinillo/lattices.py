from qmtk._lattice import *
from qmtk.utils.preprocess import alias, require


class LatticeBase(object):
    """lattice base
    """

    @alias(shape=['length', 'size'])
    @require('shape')
    def __init__(self, **kwargs):
        super(LatticeBase, self).__init__()
        self._cache = {}
        self.nbr = None
        self.shape = kwargs.pop('shape')

    @property
    def cache(self):
        if self.nbr is None:
            return None
        return self._cache[self.nbr]

    @cache.setter
    def cache(self, value):
        self._cache[self.nbr] = value

    def grid(self, nbr=None):
        if nbr is None:
            self.nbr = 0
        else:
            self.nbr = nbr

        if self.nbr not in self._cache:
            if self.nbr == 0:
                self.cache = self.sites()
            else:
                self.cache = self.bonds(self.nbr)
        return self.cache


class Chain(LatticeBase, ChainBase):
    """chain lattice
    """
    pass


class Square(LatticeBase, SquareBase):
    """square lattice
    """
    pass


class PBCChain(LatticeBase, PBCChainBase):
    """chain lattice (periodic boundary)
    """
    pass


class PBCSquare(LatticeBase, PBCSquareBase):
    """square lattice (periodic boundary)
    """
    pass
