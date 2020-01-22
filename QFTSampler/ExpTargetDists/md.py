import numpy as np
from .BaseTarget import BaseTarget

class Target_LJ2(BaseTarget):
    def __init__(self,N,M,sample_num=None):
        self.be4 = 4
        super(Target_LJ2, self).__init__(N, M, 6, sample_num)

    def __call__(self,*arg):
        def LJ(r):
            _r = r+1e-6
            return _r**-12 - _r**-6
        sample_num = len(arg[0])
        x1,y1,z1,x2,y2,z2 = [ it/(2**self.N)*2 -1 for it in arg ]
        rr = np.sqrt( np.square(x1-x2) + np.square(y1-y2) + np.square(z1-z2) )
        p = np.exp(-self.be4*LJ(rr)) / self.Z
        return p
