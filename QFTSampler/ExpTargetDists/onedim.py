import numpy as np

class Target_gauss_flat:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        r = (x**2)
        d = np.abs(r*2)
        p = np.exp(-d)
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i]] for i in range(sample_num)])

class Target_gauss_sharp:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        r = (x**2)
        d = np.abs(r*64)
        p = np.exp(-d)
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i]] for i in range(sample_num)])

class Target_gauss_multi:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        x1 = x+0.5
        x2 = x-0.25
        r1 = (x1**2)
        r2 = (x2**2)
        p = 2*np.exp(-r1*32) + np.exp(-r2*8)
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i]] for i in range(sample_num)])

class Target_gauss_multi2:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        x1 = x+0.5
        x2 = x-0.5
        r0 = (x**2)
        r1 = (x1**2)
        r2 = (x2**2)
        p = np.exp(-r1*16) + np.exp(-r2*16) + np.exp(-r0*32)*4
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i]] for i in range(sample_num)])

class Target_step:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        d = np.abs(x)
        p = np.where( d < 0.5 , 1. , 0. )
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i]] for i in range(sample_num)])
