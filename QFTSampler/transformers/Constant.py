import numpy as np

from .BaseTransformer import *
from .Standardizer import Standardization
from .Momentum import Momentum

class Constant(BaseTransformer):
    def __init__(me, M, save_params=False):
        me.M = M
        me.param = np.zeros([2**me.M], dtype = np.complex128)
        me.param[0] = 1. + 0.j
        me.param/= np.sum(np.square(np.abs(me.param)))**0.5
        me.momentum_param = Momentum( me.param )
        me.stan = Standardization()
        if save_params:
            me.param_list = []
            me.i = 0
            me.I = 100
    def phi(me,*arg,sample_num=1):
        return me.stan(np.array([me.param]*sample_num))

    def update(me,grad_phi,*arg, lr = 10):       # grad_phi = samplex(2**M) matrix
        grad = me.stan.backward(grad_phi)
        grad_phi = np.sum(grad_phi,axis=0)
        me.param -= lr* me.momentum_param(grad_phi)

        if hasattr(me, 'i'):
            if ( me.i % me.I == 0):
                me.param_list.append( me.param.copy() )
            me.i+=1

    def clear(self, ):
        self.stan.clear()
#
# class Constant(BaseTransformer):
#     def __init__(me, M):
#         me.M = M
#         me.param = np.zeros([2**me.M], dtype = np.complex128)
#         me.param[0] = 1. + 0.j
#         me.param/= np.sum(np.square(np.abs(me.param)))**0.5
#         me.stan = Standardization()
#
#     def phi(me,*arg,sample_num=1):
#         return me.stan(np.array([me.param]*sample_num))
#
#     def update(me,grad_phi,*arg, lr = 10):       # grad_phi = samplex(2**M) matrix
#         grad = me.stan.backward(grad_phi)
#         grad = np.sum(grad,axis=0)
#         me.param -= lr*grad
#         # me.param/= np.sum(np.square(np.abs(me.param)))**0.5
#
#     def clear(self, ):
#         self.stan.clear()
