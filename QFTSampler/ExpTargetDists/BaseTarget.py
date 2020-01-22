import numpy as np

class BaseTarget():
    def __init__(self, N, M, dim, sample_num=None):
        self.N = N
        self.M = M
        self.dim = dim
        self.Z = 1
        if sample_num is None:
            sample_num = min(1000000, (2**N)**dim)
        if sample_num!=0:
            self.check(sample_num=sample_num)
    def __call__(self, *arg):
        raise NotImplementedError()
    def check(self, sample_num):
        sn = sample_num
        samples = [ np.random.randint( 2**self.N ,size=(sn,)) for i in range(self.dim)]
        p = self(*samples)
        self.Z = np.sum(p)/(sn/(2**(self.dim*self.N)))
