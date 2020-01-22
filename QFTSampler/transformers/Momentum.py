import numpy as np
class Momentum():
    def __init__(me, param):
        me.momentum = np.zeros_like( param )
        me.flag = True
    def __call__(me, grad):
        if ( me.flag ):
            me.momentum = grad
            me.flag = False
        if ( np.linalg.norm(me.momentum) * 100. > np.linalg.norm(grad) ):
            me.momentum = me.momentum * 0.9 + grad * 0.1
        return me.momentum
