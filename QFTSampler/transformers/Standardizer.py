import numpy as np

class Standardization():
    # 量子コンピュータの規格化用クラス 学習は行わない
    def __init__(self, ):
        self.x = None
        self.y = None
        self.s = None
    def __call__(self, phi):#x : (N, 2^m)
        self.x = np.concatenate([phi.real,phi.imag],axis=1)
        self.L = phi.real.shape[1]
        self.s = ((np.abs(self.x)**2).sum(axis=1)**0.5).reshape(-1, 1)
        self.y = self.x/self.s
        return self.y[:,:self.L] + self.y[:,self.L:]*1j
    def backward(self, grad_phi):
        grad = np.concatenate([grad_phi.real,grad_phi.imag],axis=1)
        mat = np.einsum("bi,bj->bij",self.y,self.y)
        mat = np.expand_dims(np.eye(self.L*2),axis=0) - mat
        d1 = np.einsum("bij,bj->bi",mat,grad)
        d1 = d1 / self.s
        return d1[:,:self.L] + d1[:,self.L:]*1j

    def clear(self, ):
        self.x = None
        self.y = None
        self.s = None
