import numpy as np
from . import qSim_numpy as qsn

class QFTSampler:
    def __init__(self):
        pass
    def normalizing(self, phi, ):
        return phi/np.sum(np.abs(phi)**2)**0.5
    def ft_W(self, N, M, x):
        return np.exp(-1j*2*np.pi*x*np.arange(2**M)/2**N)

    def q(self, N, M, x, phi):
        phi = self.normalizing(phi)
        W = self.ft_W(N,M,x)
        return np.abs(W@phi)**2/(2**N)
    def q_for_samples(self, N, M, samples, phis): # samples(# of sample,)
        prob = []
        for x,phi in zip(samples,phis):
            p = self.q(N,M,x,phi)
            prob.append(p)
        return np.array(prob)

    def div_q(self, N, M, x, phi):
        assert phi.ndim == 1
        phi = self.normalizing(phi)
        phi = phi.reshape(-1, 1)
        W = self.ft_W(N,M,x)
        return 2*(W@phi)*np.conjugate(W)/(2**N)
    def div_q_for_samples(self, N, M, samples, phis): #return C(# of sample,2^M)
        div = []
        for x,phi in zip(samples,phis):
            d = self.div_q(N,M,x,phi)
            div.append(d)
        return np.array(div)

    def sample(self, N, M, phis):
        sampled = []
        phis = phis/np.sum(np.abs(phis)**2, axis=1, keepdims=True)**0.5
        for phi in phis:
            #phi = self.normalizing(phi)
            tmp = qsn.sample(ITER=1, N=N, M=M, low_state=phi, verbose=False)[0]
            sampled.append(tmp)
        sampled = np.array(sampled)
        return sampled
