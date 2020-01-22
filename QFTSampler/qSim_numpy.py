import numpy as np
import sys

verbose = 1

class Bar: #easy progress bar
    def __init__(me,N, verbose=True):
        me.N = N
        me.P = 0
        me.verbose = verbose
    def __call__(me,i):
        if me.verbose:
            p = (i*100)//(me.N-1)
            if p != me.P:
                me.P = p
                k = p//10
                j = p%10
                mes = "#"*(k) + (str(j)if i+1!=me.N else "") + "-"*max(-1,10-k-1)
                print ("\r"+mes,end="",file=sys.stderr)

class Vstate():
    def __init__(me,M,N,v=None):
        me.v = np.zeros([2**(M+1)],np.complex128) if v is None else v # [M,M-1~0]--> low_bit
        me.M = M
        me.N = N
    def copy(me):
        return Vstate(me.M,me.N,me.v.copy())
    def write(me,index,value):
        assert index < 2**me.M
        me.v[index] = value
    def renormalize(me):
        s = np.sum(np.abs(me.v)**2)
        me.v /= (s**0.5)
    def H(me,q):
        #print (me.M-1-q,q)
        v_ = me.v.reshape(1<<(me.M-q),1<<1,1<<q)
        isq2 = 1/np.sqrt(2)
        v_[:,0,:],v_[:,1,:] = (v_[:,0,:]+v_[:,1,:])*isq2,(v_[:,0,:]-v_[:,1,:])*isq2

    def degenerate(me,q):
        p = 0

        v_ = me.v.reshape(1<<(me.M-q),1<<1,1<<q)
        p = np.sum(np.abs(v_[:,0,:])**2)

        if np.random.rand() < p:
            v = 0
        else:
            v = 1

        v_[:,0,:] = v_[:,v,:]
        v_[:,1,:] = 0.

        me.renormalize()

        return v
    def Rotate(me,q,rad):

        v_ = me.v.reshape(1<<(me.M-q),1<<1,1<<q)
        v_[:,1,:] *= np.exp( -rad*1j )

        if False:
            for i in range(1<<(1+me.M)):
                index = i
                if (i&(1<<q))>0:
                    me.v[ index ] *= np.exp( -rad*1j )

    def get_whole_state(me):
        res = np.zeros([2**me.N],dtype=np.complex)
        res[:2**(me.M+1)] = me.v
        return res

def sample(ITER,M,N,low_state=[1.+0.j], verbose=True):
    res_li = []

    state_target = Vstate(M,N)
    wri = low_state
    for i in range(len(wri)):
        state_target.write(i,wri[i])
    state_target.renormalize()

    def get():
        bar = Bar(ITER, verbose=verbose)
        for i in range(ITER):
            bar(i) if verbose > 0 else None

            state = state_target.copy()

            res =[None for i in range(N)]
            for i in range(N-1,-1,-1):
                ii = i if i<M else M
                state.H(ii)
                v= state.degenerate(ii)
                res[i]=v
                for j in range(i):
                    tar = i-j-1
                    if tar<M:
                        if v==1:
                            state.Rotate(tar,np.pi/(2**j)/2)
            res_li.append( sum([ res[N-1-i]*(2**i) for i in range(N)]) )
        return res_li

    return get()
