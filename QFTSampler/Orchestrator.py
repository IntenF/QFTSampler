import numpy as np
import matplotlib.pyplot as plt
import pickle

from .QFTSampler import QFTSampler

class Orchestrator:
    def __init__(self, N, M, transformer_list, target , ):
        self.QFTSampler = QFTSampler()
        self.M = M
        self.N = N
        self.transformer_list = transformer_list
        self.target = target

    def step(self,train=True,sample_num = 64,lr=1, loss_func='CE', ):
        phis_list = []
        samples_list = []
        qs_list = []
        divqs_list = []

        for transformer in self.transformer_list:
            phis_list.append( transformer.phi(*samples_list,sample_num=sample_num) )
            samples_list.append( self.QFTSampler.sample(self.N,self.M,phis_list[-1]) )
            qs_list.append( self.QFTSampler.q_for_samples(self.N,self.M,samples_list[-1],phis_list[-1]) )
            divqs_list.append( self.QFTSampler.div_q_for_samples(self.N,self.M,samples_list[-1],phis_list[-1]) )

        q = qs_list[0].copy()
        for i in range( 1,len(self.transformer_list) ):
            q *= qs_list[i]
        #q = np.maximum( q, 1/((2**self.N)**2) * 100. )
        #q = q * 0. + 1/((2**self.N)**2)

        if train:
            #scale = (2**self.N)**len(self.transformer_list)/sample_num
            p = self.target(*samples_list)
            z_p = 1#np.sum(p)
            z_q = 1#np.sum(q)
            if loss_func=='CE':
                grad = -self.target(*samples_list)/q/z_p #旧ロス 交差エントロピーH(P|Q)~=KL(P||Q)
            elif loss_func=='KL':
                grad = (np.log(q)-np.log(p)-np.log(z_q)+np.log(z_p)+1)  #KL(Q||P)
            else:
                raise ValueError(f'not difined loss_func name {loss_func}')
            grad /= sample_num
            # grad /= scale

            grad = grad.reshape(-1, 1)
            for i in range( len(self.transformer_list) ):
                axis_grad = grad/qs_list[i].reshape(-1,1)*divqs_list[i]
                self.transformer_list[i].update(axis_grad,*(samples_list[:(i+1)]),lr=lr)

        return (q, np.array(samples_list).T,)

    def save(self, filename):
        for t in self.transformer_list:
            t.clear()
        with open(filename, mode='wb') as wh:
            pickle.dump(self, wh)

    def pmap(self, stride=16):
        dim = len(self.transformer_list)
        target = self.target
        dst_list = [np.arange(0, 2**self.N,stride) for i in range(dim)]
        target_samples_list = [ each.flatten() for each in np.meshgrid(*dst_list) ]
        p = target(*target_samples_list )
        return p.reshape(*([len(dst_list[0]), ]*dim))

    def qmap(self, stride=16):
        dim = len(self.transformer_list)
        dst_list = [np.arange(0, 2**self.N,stride) for i in range(dim)]
        target_samples_list = [ each.flatten() for each in np.meshgrid(*dst_list) ]

        sample_num = len(dst_list[0])**dim
        phis_list = []
        samples_list = []
        qs_list = []
        for i,transformer in enumerate(self.transformer_list):
            phis_list.append( transformer.phi(*samples_list,sample_num=sample_num) )
            #samples_list.append( self.QFTSampler.sample(self.N,self.M,phis_list[-1]) )
            samples_list.append( target_samples_list[i] ) # force to sample 'target_samples'
            qs_list.append( self.QFTSampler.q_for_samples(self.N,self.M,samples_list[-1],phis_list[-1]) )
        return np.prod([ qs.reshape(*[len(dst_list[0]), ]*dim) for qs in qs_list ], axis=0)



def load_orchestrator(filename):
    with open(filename, mode='rb') as rh:
        tmp = pickle.load(rh)
    return tmp
