from enum import Enum
from abc import ABC, abstractmethod
import random
from collections import defaultdict
import math
class Layer(ABC):
    def __init__(self, name):
        self.name = name
        self.grads = defaultdict(dict)
        
    @abstractmethod
    def init_parameter(self):
        pass
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def backward(self):
        pass
    
    @abstractmethod
    def step(self):
        pass
    
class LinearLayer(Layer):
    def __init__(self, name, d_in, d_out):
        super(LinearLayer, self).__init__(name)
        self.d_in = d_in
        self.d_out = d_out
        self.init_parameter()
        
    def init_parameter(self):
        self.W = [[random.normalvariate(0, 0.5) for _ in range(self.d_in)] for _ in range(self.d_out)]
        self.b = [[0.0 for _ in range(self.d_out)]]
    
    def forward(self, x):
        BS = len(x)
        y = [[0 for _ in range(self.d_out)] for _ in range(BS)]
        for i in range(BS):
            for j in range(self.d_out):
                tmp = 0
                for k in range(self.d_in):
                    tmp += x[i][k] * self.W[j][k]
                y[i][j] = tmp

        for i in range(BS):
            for j in range(self.d_out):
                y[i][j] += self.b[0][j]
        return y
    
    def backward(self, dLdy, cache):
        # dLdy has shape [BS, d_out]
        BS = len(dLdy)
        x = cache["in"]
        dLdW = [[0 for _ in range(self.d_in)] for _ in range(self.d_out)]
        for i in range(self.d_out):
            for j in range(self.d_in):
                tmp = 0
                for k in range(BS):
                    tmp += x[k][j] * dLdy[k][i]
                dLdW[i][j] = tmp / BS
        dLdb = [[0.0 for _ in range(self.d_out)]]
        for i in range(self.d_out):
            for j in range(BS):
                dLdb[0][i] += dLdy[j][i] / BS
        self.grads["dW"] = dLdW
        self.grads["db"] = dLdb
        dLdx = [[0 for _ in range(self.d_in)] for _ in range(BS)]
        for i in range(BS):
            for j in range(self.d_in):
                tmp = 0
                for k in range(self.d_out):
                     tmp += self.W[k][j] * dLdy[i][k]
                dLdx[i][j] = tmp
        return dLdx
    
    def step(self, lr):
        for i in range(self.d_out):
            for j in range(self.d_in):
                self.W[i][j] -= lr * self.grads["dW"][i][j]
        for i in range(self.d_out):
            self.b[0][i] -= lr * self.grads["db"][0][i]
            
class ReLuLayer(Layer):
    def __init__(self, name):
        super(ReLuLayer, self).__init__(name)
    def init_parameter(self):
        pass
    def forward(self, x):
        BS = len(x)
        d = len(x[0])
        y = [[0 for _ in range(d)] for _ in range(BS)]
        for i in range(BS):
            for j in range(d):
                y[i][j] = max(0, x[i][j])
        return y
    def backward(self, dLdy, cache):
        BS = len(dLdy)
        d = len(dLdy[0])
        x = cache["in"]
        dLdx = [[0 for _ in range(d)] for _ in range(BS)]
        for i in range(BS):
            for j in range(d):
                dLdx[i][j] = 0 if x[i][j] <= 0 else dLdy[i][j]
        return dLdx
    def step(self, lr):
        pass 

class SigmoidLayer(Layer):
    def __init__(self, name):
        super(SigmoidLayer, self).__init__(name)
    def init_parameter(self):
        pass
    def forward(self, x):
        BS = len(x)
        d = len(x[0])
        y = [[0 for _ in range(d)] for _ in range(BS)]
        for i in range(BS):
            for j in range(d):
                y[i][j] = 1.0 / (1 + math.exp(-x[i][j]))
        return y
    def backward(self, dLdy, cache):
        BS = len(dLdy)
        d = len(dLdy[0])
        y = cache["out"]
        dLdx = [[0 for _ in range(d)] for _ in range(BS)]
        for i in range(BS):
            for j in range(d):
                dLdx[i][j] = dLdy[i][j] * y[i][j] * (1.0 - y[i][j])
        return dLdx
    def step(self, lr):
        pass 

class Loss(ABC):
    def __init__(self, name):
        self.name = name
    @abstractmethod
    def forward(self, predicted, target):
        pass
    @abstractmethod
    def backward(self, L):
        pass

class BCELoss(Loss):
    def __init__(self, name):
        super(BCELoss, self).__init__(name)
    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        # predicted [BS, 1]
        # target [BS]
        BS = len(target)
        L = 0.0
        for i in range(BS):
            if target[i] == 1:
                L += - math.log(predicted[i][0]) / BS
            else:
                L += - math.log(1.0 - predicted[i][0]) / BS
        return L
    def backward(self):
        BS = len(self.target)
        dL = [[0.0] for _ in range(BS)]
        for i in range(BS):
            dL[i][0] = (self.predicted[i][0] - self.target[i]) / (self.predicted[i][0]*(1-self.predicted[i][0]))
        return dL

class MSELoss(Loss):
    def __init__(self, name):
        super(MSELoss, self).__init__(name)
    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        BS = len(target)
        L = 0.0
        for i in range(BS):
            L += (predicted[i][0] - target[i]) ** 2 / BS
        return L
    def backward(self):
        BS = len(self.target)
        dL = [[0.0] for _ in range(BS)]
        for i in range(BS):
            dL[i][0] = 2 * (self.predicted[i][0] - self.target[i])
        return dL
    
class Model:
    def __init__(self):
        self.layers = []
        self.cache = defaultdict(dict)
        self.loss = BCELoss("bce")
        self.lr = 1e-1
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def forward(self,x):
        for l in self.layers:
            self.cache[l.name]['in'] = x
            x = l.forward(x)
            self.cache[l.name]['out'] = x
        return x

    def backward(self, target):
        last_layer = self.layers[-1]
        predicted = self.cache[last_layer.name]['out']
        loss = self.loss.forward(predicted, target)
        dL = self.loss.backward()
        for l in reversed(self.layers):
            dL = l.backward(dL, self.cache[l.name])
        for l in self.layers:
            l.step(lr=self.lr)
        return loss
    
    def train_one_epoch(self, x, y):
        predict = self.forward(x)
        loss = self.backward(y)
        return loss, predict
    
    def train(self, epoch, x, y):
        for e in range(epoch):
            loss, predict = self.train_one_epoch(x, y)
            print(loss, predict)
                
model = Model()
model.add_layer(LinearLayer("linear1", 3, 4))
model.add_layer(SigmoidLayer("sigmoid1"))
model.add_layer(LinearLayer("linear2", 4, 1))
model.add_layer(SigmoidLayer("sigmoid2"))

x = [[-1,-1,-1],[1,1,1]]
y = [1,0]
model.train(500,x,y)
