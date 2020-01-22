from .BaseTarget import BaseTarget
from ..transformers import Constant,Affine
from ..Orchestrator import Orchestrator
import numpy as np

from scipy import signal
class ANPAN(BaseTarget):
    def __init__(self, N, M, size=2**10, kernel_size=None, kernel_type='gaussian', sample_num=None):
        x = np.arange(size)
        y = np.arange(size)
        xx, yy = np.meshgrid(x, y)
        p = self.init_p(*[xx.flatten(),yy.flatten()]).reshape(size, -1)

        #kernel
        if kernel_size is None:
            kernel_size = 2**6+1
        if kernel_size != 1:
            size = kernel_size
            if kernel_type == 'gaussian':
                if size%2==0:
                    print('kernel size should be odd')
                    return
                sigma = (size-1)/2
                # [0,size]→[-sigma, sigma] にずらす
                x = y = np.arange(0,size) - sigma
                X,Y = np.meshgrid(x,y)
                mat = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
                # 総和が1になるように
                kernel = mat / np.sum(mat)
            elif kernel_type == 'ma':
                kernel = np.ones((size, size))/(size**2)

            p = signal.convolve2d(p, kernel, mode='same', )

        self.p = p
        super().__init__(N, M, 2, sample_num)

    def init_p(self, *arg):
        self.N = 10
        x = (np.array(arg[0])/(2**self.N)*2-1)*110
        y = (np.array(arg[1])/(2**self.N)*2-1)*110
        c_list = []
        c_list.append((x/85)**2 +(y/85)**2 -1)
        c_list.append(((abs(x)-55)/24)**2 + ((y+5)/24)**2 -1)
        c_list.append(((x/28)**2 +((y+5)/23)**2) -1)
        c_list.append(((abs(x)-18)/9)**2 +((y-40)/15)**2 -1)
        c_list.append((np.sign(y-45) +1)/2*(((abs(x)-18)/12)**2 +((y-45)/20)**2 -2) +1)
        c_list.append(((np.sign(-y-35)+1)/2)*((x/42)**2+((y+35)/18)**2 -2) +1)
        c_list.append(-np.maximum(np.abs((x)/110), np.abs((y)/110))**3 + 1.5)
        s = 1
        epsilon = 1e-4
        for c in c_list:
            _c = np.abs(c).clip(0+epsilon,1)
#             _c = np.where(c<0, -c, c)
#             _c = np.where(_c>1, 1, _c)
            s *= _c
        s *= (-((x/85)**2 +(y/85)**2 -1)).clip(0, 1)
        return s

    def __call__(self, *arg):
        x = (np.array(arg[0])/(2**self.N)*(2**10)).astype(np.int)
        y = (np.array(arg[1])/(2**self.N)*(2**10)).astype(np.int)
        s = self.p[-y, x]
        return s/self.Z

class Target_gauss2d_independent:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        y = np.linspace(-1,1,2**N)
        x,y = np.meshgrid(x,y)
        d = 2 * (x**2 + y**2)
        p = np.exp(-d)
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i],arg[1][i]] for i in range(sample_num)])

class Target_gauss2d_dependent:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        y = np.linspace(-1,1,2**N)
        x,y = np.meshgrid(x,y)
        c,s = np.cos(np.pi/4),np.sin(np.pi/4)
        x,y = x*c+y*s,x*s-y*c
        d = 2 * (x**2 + 4*(y**2))
        p = np.exp(-d)
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i],arg[1][i]] for i in range(sample_num)])

class Target_gauss2d_dependent:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        y = np.linspace(-1,1,2**N)
        x,y = np.meshgrid(x,y)
        c,s = np.cos(np.pi/4),np.sin(np.pi/4)
        x,y = x*c+y*s,x*s-y*c
        d = 2 * (x**2 + 4*(y**2))
        p = np.exp(-d)
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i],arg[1][i]] for i in range(sample_num)])

class Target_gauss2d_multi:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.arange(0, 2**N)
        y = np.arange(0, 2**N)
        X, Y = np.meshgrid(x, y)
        self.scale = np.sum(self.prob(X.reshape(-1, 1), Y.reshape(-1, 1)))

    def prob(self, *arg):
        x_samples = arg[0]
        y_samples = arg[1]
        p = np.zeros([len(x_samples)])
        for i in range(len(x_samples)):
            x = ( x_samples[i] / 2**self.N ) *2. -1
            y = ( y_samples[i] / 2**self.N ) *2. -1
            s0 = (x-0.5)+(y+0.5)
            t0 = (x-0.5)-(y+0.5)
            s1 = (x+0.5)+(y-0.5)
            t1 = (x+0.5)-(y-0.5)
            p[i] = np.exp( -(s0*s0+4*t0*t0) )/2+ np.exp( -(s1*s1+4*t1*t1) )/2
        return p

    def __call__(self, *arg):
        p = self.prob(*arg)
        p = np.array(p)/self.scale
        return p

class Target_gauss2d_multi:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        y = np.linspace(-1,1,2**N)
        x,y = np.meshgrid(x,y)
        x1,y1 = x+0.5,y+0.5
        x2,y2 = x-0.5,y-0.5
        d1 = 8 * (x1**2 + (y1**2))
        d2 = 8 * (x2**2 + (y2**2))
        p = np.exp(-d1) + np.exp(-d2)
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i],arg[1][i]] for i in range(sample_num)])

class Target_circle2d:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        y = np.linspace(-1,1,2**N)
        x,y = np.meshgrid(x,y)
        d = ( np.sqrt(x**2 + y**2) -1/2) ** 2
        #p = np.exp(-d)
        p = np.exp(-d*32)
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i],arg[1][i]] for i in range(sample_num)])

class Target_checker_booard:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        x = np.linspace(-1,1,2**N)
        y = np.linspace(-1,1,2**N)
        x,y = np.meshgrid(x,y)
        d = x*y
        p = np.where( d > 0. , 1. , 0. )
        p/=np.sum(p)
        self.p = p

    def __call__(self,*arg):
        sample_num = len(arg[0])
        return np.array([self.p[arg[0][i],arg[1][i]] for i in range(sample_num)])

class Target_linear2d:
    def __init__(self,N,M):
        self.N = N
        self.M = M
        transformer_list = [ Constant(M) ,  Affine(N,M) ]
        con = transformer_list[0]
        aff = transformer_list[1]

        aff.w *= 0.
        aff.b *= 0.
        aff.b[0] = 1.
        aff.w[0,1] = 1.
        aff.w[0,2] = .2
        aff.w[0,2+2**M] = .4
        aff.w[0,3] = .5

        aff.b[1] = 0.
        aff.b[3] = 1.

        con.param *= 0.
        con.param[0] = 1.
        con.param /= np.sum(np.square(np.abs(con.param)))**0.5

        if True:
            con.param[0] = 3
            con.param[1] = 1 + 1j
            con.param[2] = 2 + 2j
            con.param[3] = 3 + 3j
            con.param /= np.sum(np.square(np.abs(con.param)))**0.5

        orch= Orchestrator(N,M,transformer_list,None,)
        self.QFTSampler = orch.QFTSampler
        self.transformer_list = orch.transformer_list
        x = np.arange(2**N)
        y = np.arange(2**N)
        x,y = np.meshgrid(x,y)
        #d = x*y
        #p = np.where( d > 0. , 1. , 0. )
        #p/=np.sum(p)
        #self.p = p

    def __call__(self,*arg):
        target_samples_list = arg
        sample_num = len(arg[0])
        phis_list = []
        samples_list = []
        qs_list = []
        divqs_list = []
        for i,transformer in enumerate(self.transformer_list):
            phis_list.append( transformer.phi(*samples_list,sample_num=sample_num) )
            #samples_list.append( self.QFTSampler.sample(self.N,self.M,phis_list[-1]) )
            samples_list.append( target_samples_list[i] ) # force to sample 'target_samples'
            qs_list.append( self.QFTSampler.q_for_samples(self.N,self.M,samples_list[-1],phis_list[-1]) )
            divqs_list.append( self.QFTSampler.div_q_for_samples(self.N,self.M,samples_list[-1],phis_list[-1]) )
        p = (qs_list[0]*qs_list[1])
        return p
