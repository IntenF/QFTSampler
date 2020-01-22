import numpy as np
from .BaseTransformer import BaseTransformer
from .Standardizer import Standardization

import numpy as np

class SGD:
    def __init__(self, variables, lr=None, decay=0.,):
        self.lr = lr
        self.variables = variables
        self.decay = decay
    def update(self, lr=None, ):
        if lr is None:
            assert self.lr is not None
            lr = self.lr
        for v in self.variables:
            v.var -= lr*(v.grad + v.var*self.decay) #with L2 norm decay

class Momentum():
    def __init__(self, variables, lr=None, decay=0., momentum=0.9):
        self.lr = lr
        self.variables = variables
        self.decay = decay
        self.alpha = momentum
    def __call__(self, ):
        self.update()

    def update(self, lr=None, ):
        if lr is None:
            assert self.lr is not None, f'you didnot give lr for {self.__class__.__name__}.update(). you must give {self.__class__.__name__} lr on __init__ or update()'
            lr = self.lr

        for v in self.variables:
            if not hasattr(v, 'momentum'):
                v.momentum = v.grad
        if ( np.linalg.norm(v.momentum) * 100. > np.linalg.norm(v.grad) ):
            a = self.alpha
            v.momentum = v.momentum*a + (1-a)*v.grad

        for v in self.variables:
            v.var -= lr*v.momentum


class Variable:
    def __init__(self, init_var, ):
        self.var = init_var
        self.zero_grad()
    def zero_grad(self, ):
        self.grad = np.zeros(self.var.shape)

class BaseLayer:
    def __init__(self, activation=None, input_layers=set()):
        self.input_layers = input_layers
    def trainables(self, ):
        '''
        return: list of Variables
        '''
        raise NotImplementedError()
    def __call__(self, *args, ):
        return self.forward(*args)
    def forward(self, inputs, ):
        raise NotImplementedError()
    def backward(self, grad, ):
        raise NotImplementedError()
    def zero_grad(self,):
        for v in self.trainables():
            v.zero_grad()
    def clear(self,):
        pass

class IdentityLayer(BaseLayer):
    def __init__(self, **argv):
        super().__init__(**argv)
    def trainables(self, ):
        return []
    def forward(self, x, ):
        return x
    def backward(self, grad, ):
        return grad


class DenseLayer(BaseLayer):
    def __init__(self, in_unit, out_unit, activation=None, init='He', **argv):
        super().__init__(**argv)
        self.in_unit = in_unit
        self.out_unit = out_unit
        self.w = Variable(np.random.randn(in_unit, out_unit ).astype(np.float64))
        #self.w.var = np.eye(in_unit, out_unit, dtype=np.float64)
        self.b = Variable(np.zeros([out_unit ],dtype=np.float64))
        #self.b.var[0] = 1.
        self.x = None
        if activation is not None:
            self.activation = activation
            if isinstance(activation, Sigmoid):
                self.w_scale = (1/in_unit)**0.5
            elif isinstance(activation, Relu):
                self.w_scale = (2/in_unit)**0.5
        else:
            self.activation = None
            self.w_scale = 1

    def trainables(self, ):
        return [self.w, self.b]
    def forward(self, x, ):
        '''
        x (batch, in_unit)
        '''
        #self.b.var /= np.linalg.norm(self.b.var, ord=2, )
        self.x = x
        y = x@(self.w.var*self.w_scale) +self.b.var
        return y if self.activation is None else self.activation.forward(y)

    def backward(self, grad, ):       # grad_phi = samplex(2**out_unit) matrix
        '''
        grad (batch, out_unit)
        '''
        x = self.x
        grad = grad if self.activation is None else self.activation.backward(grad)
        x_grad = grad@(self.w.var*self.w_scale).T#(batch, in_unit)
        self.w.grad += (x.T@grad)/self.w_scale#/len(grad)
        self.b.grad += np.sum(grad, axis=0)#/len(grad)
        return x_grad

    def clear(self,):
        self.x = None

class Sigmoid(BaseLayer):
    def __init__(self, **argv ):
        super().__init__(**argv)
        self.y = None
    def trainables(self, ):
        return []
    def forward(self, x, ):
        '''
        x (batch, dim)
        '''
        y = 1/(1+np.exp(-x))
        self.y = y
        return y
    def backward(self, grad, ):
        '''
        grad (batch, dim)
        '''
        y = self.y
        return grad*y*(1-y)

    def clear(self,):
        self.y = None

class Relu(BaseLayer):
    def __init__(self, **argv ):
        super().__init__(**argv)
        self.filter = None
    def trainables(self, ):
        return []
    def forward(self, x, ):
        self.filter = x>=0
        return np.where(self.filter, x, 0)
    def backward(self, grad, ):
        return np.where(self.filter, grad, 0)
    def clear(self,):
        self.filter=None

class ConstantLayer(BaseLayer):
    def __init__(self, dim, **argv ):
        super().__init__(**argv)
        self.const = Variable(np.zeros(dim, dtype=np.float64))
        self.const.var[0] = 1.
    def trainables(self, ):
        return [self.const]
    def forward(self, x, ):
        return np.tile(self.const.var[np.newaxis, :], (len(x), 1))
    def backward(self, grad, ):
        self.const.grad += np.sum(grad, axis=0,)
        return None

class FiexedConstantLayer(BaseLayer):
    def __init__(self, dim, const=None, **argv):
        super().__init__(**argv)
        self.const = Variable(np.zeros(dim, dtype=np.float64))
        if const is None:
            self.const.var[0] = 1.
        else:
            assert len(const) == dim, f'dim({dim}) is not equal to const dimension({len(const)})'
            self.const.var = const
    def trainables(self, ):
        return []
    def forward(self, x, ):
        return np.tile(self.const.var[np.newaxis, :], (len(x), 1))
    def backward(self, grad, ):
        return None

class Seaquence:
    def __init__(self, layers, ):
        self.layers = layers
        self.bs = None
    def __call__(self, *args):
        return self.forward(*args)
    def forward(self, x, ):
        self.bs = len(x)
        for l in self.layers:
            x = l.forward(x)
        return x
    def backward(self, grad, ):
        g = grad
        for i,l in enumerate(self.layers[::-1]):
            g = l.backward(g)
    def zero_grad(self, ):
        for l in self.layers[::-1]:
            g = l.zero_grad()

    def clear(self, ):
        for l in self.layers:
            g = l.clear()

    def trainables (self, ):
        trainv = []
        for l in self.layers:
            trainv += l.trainables()
        return trainv

    def numerical_diff(self, x, delta=0.001):
        y = self.forward(x)[0][0]
        for tmpv in self.trainables():
            v = tmpv.var
            g = tmpv.grad
            shape = v.shape
            assert len(shape) < 3, f'次元数は3以下のものは数値微分できません. {l}の{tmpv}は次元数が{len(shape)}です'
            if len(shape) == 1:
                for i in range(shape[0]):
                    v[i] += delta
                    y_ = self.forward(x)[0][0]
                    g[i] += (y_-y)/delta
                    v[i] -= delta

            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        v[i][j] += delta
                        y_ = self.forward(x)[0][0]
                        g[i][j] += (y_-y)/delta
                        v[i][j] -= delta

class Model:
    def __init__(self, inputs, outputs):
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
        add_layers = self.outputs
        forward_connections = {}
        while True:
            next_add_layers = []
            for l in add_layers:
                if len(l.input_layers) == 0:
                    assert l in self.inputs, f'root layer({l}) must be in {inputs}'
                for ll in l.input_layers:
                    if ll not in forward_connections.keys():
                        next_add_layers.append(ll)
                        forward_connections[ll] = set()
                    forward_connections[ll] |= {l}
            if len(next_add_layers) == 0:
                break
            add_layers = next_add_layers
        self.layers = set(forward_connections.keys()) | set(self.outputs)
        self.forward_connections = forward_connections
    def __call__(self, *args):
        return self.forward(*args)
    def forward(self, x, ):
        y = {}
        for l in self.inputs:
            y[l] = l.forward(x)
        while True:
            for l in self.layers:
                if l in y.keys():
                    continue
                if len(l.input_layers-y.keys()) == 0:
                    s = 0
                    for ll in l.input_layers:
                        s += y[ll]
                    y[l] = l.forward(s)
            if len(set(self.layers)-y.keys()) == 0:
                break
        res = 0
        for l in self.outputs:
            res += y[l]
        return res
    def backward(self, grad, ):
        y = {}
        for l in self.outputs:
            y[l] = l.backward(grad)
        while True:
            for l in self.layers:
                if l in y.keys():
                    continue
                if len(self.forward_connections[l] - y.keys()) == 0:
                    s = 0
                    c = 0
                    for ll in self.forward_connections[l]:
                        s += y[ll]
                        c += 1
                    y[l] = l.backward(s)
            if len(set(self.layers) - y.keys()) == 0:
                break
    def zero_grad(self, ):
        for l in self.layers:
            g = l.zero_grad()

    def clear(self, ):
        for l in self.layers:
            g = l.clear()

    def trainables (self, ):
        trainv = []
        for l in self.layers:
            trainv += l.trainables()
        return trainv

    def numerical_diff(self, x, delta=0.001):
        y = self.forward(x)[0][0]
        for tmpv in self.trainables():
            v = tmpv.var
            g = tmpv.grad
            shape = v.shape
            assert len(shape) < 3, f'次元数は3以下のものは数値微分できません. {l}の{tmpv}は次元数が{len(shape)}です'
            if len(shape) == 1:
                for i in range(shape[0]):
                    v[i] += delta
                    y_ = self.forward(x)[0][0]
                    g[i] += (y_-y)/delta
                    v[i] -= delta

            elif len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        v[i][j] += delta
                        y_ = self.forward(x)[0][0]
                        g[i][j] += (y_-y)/delta
                        v[i][j] -= delta


class Densenet(BaseTransformer):
    def __init__(self, N, M, in_dim, layers=None, model=None, stan=True, optimizer=SGD, ):
        self.N = N
        self.M = M
        self.stan = stan
        if layers is None and model is None:
            # layers = [DenseLayer(in_dim, 8), Relu(), DenseLayer(8, 8,), Relu(), DenseLayer(8, (2**M)*2), ]
            layers = [DenseLayer(in_dim, (2**M)*2), ]
            for l in layers:
                if type(l) is DenseLayer:
                    l.w.var *= 0
                    l.b.var *= 0
                    l.b.var[0] = 1.

            #最終層だけ初期値の分布が一様分布になるように設定する
            # layers[-1].w.var *= 0.01 #完全に０にしてしまうと逆伝搬が動かなくなる
            # layers[-1].b.var *= 0
            # layers[-1].b.var[0] = 1.

        self.seq = Seaquence(lyers) if model is None else model
        self.stan = Standardization()
        self.opt = optimizer(variables=self.seq.trainables(), )

    def phi(self, *arg, sample_num, **argv):
        if len(arg)==0:
            arg = np.empty((1, sample_num, ))
        x = np.array(arg).T #x (batch, dim)
        x = (x/2**self.N) * 2. -1.
        tmp = self.seq.forward(x)
        phi = tmp[:,:2**self.M] + tmp[:,2**self.M:] * 1j
        if self.stan:
            phi = self.stan(phi)
        return phi

    def backward(self, grad_phi, ):
        if self.stan:
            grad_phi =self.stan.backward(grad_phi)
        grad = np.concatenate([grad_phi.real,grad_phi.imag],axis=1)
        self.seq.backward(grad)

    def update(self, grad_phi,*arg, lr=1e-3):       # grad_phi = samplex(2**M) matrix
        self.backward(grad_phi)
        self.opt.update(lr)
        self.seq.zero_grad()

    def trainables(self, ):
        return self.seq.trainables()

    def clear(self, ):
        self.stan.clear()
        self.seq.zero_grad()
        self.seq.clear()
