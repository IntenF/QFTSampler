import numpy as np
from .BaseTransformer import BaseTransformer
from .Standardizer import Standardization
from .Momentum import Momentum

class AffineX(BaseTransformer):
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.w = np.zeros([2**self.M*2],dtype=np.float64)
        self.b = np.zeros([2**self.M*2])
        self.b[0] = 1.
    def phi(self,*arg, **argv):
        self.b /= np.sum(np.square(self.b))**0.5
        phi_li = []
        for i in range( len(arg[0]) ):
            x = (arg[0][i]/2**self.N) * 2. -1.
            tmp =self.w * x +self.b
            phi = tmp[:2**self.M] + tmp[2**self.M:] * 1j
            phi /= np.sum(np.square(np.abs(phi)))**0.5
            phi_li.append(phi)
        phi_li =  np.array(phi_li)
        return phi_li

    def update(self,grad_phi,*arg, lr = 1):       # grad_phi = samplex(2**M) matrix
        grad_phi_ri = np.concatenate([grad_phi.real,grad_phi.imag],axis=1)
        # sample x (2* 2**M)
        x = (arg[0]/2**self.N)*2. -1.
        self.w -= lr* np.sum( x.reshape(-1,1) * grad_phi_ri, axis=0)/len(grad_phi)
        self.b -= lr* np.sum( grad_phi_ri ,axis=0 )/len(grad_phi)

class Affine(BaseTransformer):
    def __init__(self, N, M, in_dim=1):
        self.N = N
        self.M = M
        self.w = np.zeros([in_dim, 2**self.M*2 ],dtype=np.float64)
        self.b = np.zeros([2**self.M*2])
        self.b[0] = 1.
        self.stan = Standardization()
        self.x = None

    def phi(self, *arg, **argv):
        #self.b /= np.sum(np.square(self.b))**0.5
        x = np.array(arg).T #x (batch, dim)
        x = (x/2**self.N) * 2. -1.
        self.x = x
        tmp = x@self.w +self.b
        phi = tmp[:,:2**self.M] + tmp[:,2**self.M:] * 1j
        phi = self.stan(phi)
        return phi
    def clear(self, ):
        self.stan.clear()

    def update(self, grad_phi,*arg, lr = 1):       # grad_phi = samplex(2**M) matrix
        grad_phi =self.stan.backward(grad_phi)
        grad = np.concatenate([grad_phi.real,grad_phi.imag],axis=1)
        x = self.x
        self.w -= lr* (x.T@grad)#/len(grad_phi)
        self.b -= lr*np.sum(grad, axis=0)#/len(grad_phi)

class AffineLinearBasis (BaseTransformer):
    def __init__(self, N, M, save_params=False):
        in_dim = 1
        self.N = N
        self.M = M
        self.w = 0.01+0.01*np.random.rand(in_dim, 2**self.M*2).astype(np.float64)#np.zeros([in_dim, 2**self.M*2 ],dtype=np.float64)
        self.b = np.zeros([2**self.M*2])
        self.b[0] = 1.
        self.stan = Standardization()
        self.x = None
        self.momentum_w = Momentum( self.w )
        self.momentum_b = Momentum( self.b )
        if save_params:
            self.w_list = []
            self.b_list = []
            self.i = 0
            self.I = 100

    def get_x(me,x):
        x1 = 2.*x -1.
        return np.concatenate( [x1] ,axis=1 )

    def phi(self, *arg, **argv):
        x = arg[0].reshape(-1,1)/2**self.N #(samples,in_dim)
        x = self.get_x(x)

        tmp = self.b
        phi = np.array( [tmp[:2**self.M] + tmp[2**self.M:] * 1j] * argv['sample_num'] )
        #phi (samples,dim)
        xw = x@self.w # (samples,in_dim)@(in_dim,2*dim)->(samples,2*dim)
        wphi = xw[:,:2**self.M] + xw[:,2**self.M:] * 1j #(samples,dim)
        #self.cphi = phi.copy()
        #self.wphi = wphi.copy()
        #self.x = x.copy()
        #self.ww = self.w.copy()
        phi = phi + wphi
        phi = self.stan(phi)
        return phi

    def update(self, grad_phi,*arg, lr = 1):       # grad_phi = samplex(2**M) matrix
        sample_num = len(arg[0])
        x = arg[0].reshape(-1,1)/2**self.N #(samples,in_dim=1)
        x = self.get_x(x)

        grad_phi = self.stan.backward(grad_phi)
        grad = np.concatenate([grad_phi.real,grad_phi.imag],axis=1) #(samples,2*dim)
        self.dw = x.T@grad/sample_num # (samples,in_dim).T @ (samples,2*dim) -> (in_dim,2*dim)
        self.db = np.average(grad, axis=0) #(2*dim)

        self.w -= lr* ( self.momentum_w(self.dw) )
        self.b -= lr* ( self.momentum_b(self.db) )

        if hasattr(self, 'i'):
            if ( self.i % self.I == 0 ):
                self.w_list.append( self.w.copy() )
                self.b_list.append( self.b.copy() )
            self.i += 1
    def clear(self, ):
        self.stan.clear()

class AffineNonLinearBasis (BaseTransformer):
    def __init__(self, N, M, save_params=False):
        in_dim = 4
        self.N = N
        self.M = M
        self.w = 0.01+0.01*np.random.rand(in_dim, 2**self.M*2).astype(np.float64)#np.zeros([in_dim, 2**self.M*2 ],dtype=np.float64)
        self.b = np.zeros([2**self.M*2])
        self.b[0] = 1.
        self.stan = Standardization()
        self.x = None
        self.momentum_w = Momentum( self.w )
        self.momentum_b = Momentum( self.b )
        if save_params:
            self.w_list = []
            self.b_list = []
            self.i = 0
            self.I = 100

    def get_x(me,x):
        x1 = 2.*x -1.
        x2 = np.square(x1)
        x3 = x1 ** 3
        x4 = np.sqrt(np.abs(x1))
        return np.concatenate( [x1,x2,x3,x4] ,axis=1 )

    def phi(self, *arg, **argv):
        x = arg[0].reshape(-1,1)/2**self.N #(samples,in_dim)
        x = self.get_x(x)

        tmp = self.b
        phi = np.array( [tmp[:2**self.M] + tmp[2**self.M:] * 1j] * argv['sample_num'] )
        #phi (samples,dim)
        xw = x@self.w # (samples,in_dim)@(in_dim,2*dim)->(samples,2*dim)
        wphi = xw[:,:2**self.M] + xw[:,2**self.M:] * 1j #(samples,dim)
        #self.cphi = phi.copy()
        #self.wphi = wphi.copy()
        #self.x = x.copy()
        #self.ww = self.w.copy()
        phi = phi + wphi
        phi = self.stan(phi)
        return phi

    def update(self, grad_phi,*arg, lr = 1):       # grad_phi = samplex(2**M) matrix
        sample_num = len(arg[0])
        x = arg[0].reshape(-1,1)/2**self.N #(samples,in_dim=1)
        x = self.get_x(x)

        grad_phi = self.stan.backward(grad_phi)
        grad = np.concatenate([grad_phi.real,grad_phi.imag],axis=1) #(samples,2*dim)
        self.dw = x.T@grad/sample_num # (samples,in_dim).T @ (samples,2*dim) -> (in_dim,2*dim)
        self.db = np.average(grad, axis=0) #(2*dim)

        self.w -= lr* ( self.momentum_w(self.dw) )
        self.b -= lr* ( self.momentum_b(self.db) )

        if hasattr(self, 'i'):
            if ( self.i % self.I == 0 ):
                self.w_list.append( self.w.copy() )
                self.b_list.append( self.b.copy() )
            self.i += 1
    def clear(self, ):
        self.stan.clear()
