

import torch
import numpy as np
from torch.autograd import Variable
from tme6 import CirclesData

dtype = torch.FloatTensor

def init_params(nx, nh, ny):
    params = {}
    params["Wh"] = Variable(torch.randn(nh, nx).type(dtype), requires_grad=True) * 0.3
    params["bh"] = Variable(torch.zeros(nh, 1).type(dtype), requires_grad=True)
    params["Wy"] = Variable(torch.randn(ny, nh).type(dtype), requires_grad=True) * 0.3
    params["by"] = Variable(torch.zeros(ny, 1).type(dtype), requires_grad=True)
    return params

def forward(params, X):
    bsize = X.size(0)
    nh = params['Wh'].size(0)
    ny = params['Wy'].size(0)
    outputs = {}
    outputs['X'] = X
    outputs['htilde'] = torch.mm(X, params['Wh'].t()) + params['bh'].t().expand(bsize, nh)
    outputs['h'] = torch.tanh(outputs['htilde'])
    outputs['ytilde'] = torch.mm(outputs['h'], params['Wy'].t()) + params['by'].t().expand(bsize, ny)
    outputs['yhat'] = torch.exp(outputs['ytilde'])
    outputs['yhat'] = outputs['yhat'] / (outputs['yhat'].sum(1, keepdim=True)).expand_as(outputs['yhat'])
    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = - torch.mean(Y * torch.log(Yhat))

    _, indYhat = torch.max(Yhat, 1)
    _, indY = torch.max(Y, 1)

    acc = torch.sum(indY == indYhat) * 100 / indY.size(0);

    return L, acc

def sgd(params, eta):
    params['Wy'].data -= eta * params['Wy'].grad.data
    params['Wh'].data -= eta * params['Wh'].grad.data
    params['by'].data -= eta * params['by'].grad.data
    params['bh'].data -= eta * params['bh'].grad.data
    params['Wy'].grad.data.zeros_()
    params['Wh'].grad.data.zeros_()
    params['by'].grad.data.zeros_()
    params['bh'].grad.data.zeros_()

    return params



if __name__ == '__main__':

    data = CirclesData()

    data.plot_data()

    # init
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    params = init_params(nx, nh, ny)

    curves = [[],[], [], []]

    # epoch
    for iteration in range(20):

        perm = np.random.permutation(N)
        Xtrain = data.Xtrain[perm, :]
        Ytrain = data.Ytrain[perm, :]

        # batches
        for j in range(N // Nbatch):
            indsBatch = range(j * Nbatch, (j+1) * Nbatch)
            X = Xtrain[indsBatch, :]
            Y = Ytrain[indsBatch, :]
            Yhat, outputs = forward(params, Variable(X.type(dtype), requires_grad=False))
            L, _ = loss_accuracy(Yhat, Variable(Y.type(dtype), requires_grad=False))
            L.backward()
            params = sgd(params, 0.03)

        Yhat_train, _ = forward(params, Variable(data.Xtrain.type(dtype),
            requires_grad=False))
        Yhat_test, _ = forward(params, Variable(data.Xtest.type(dtype),
            requires_grad=False))
        Ltrain, acctrain = loss_accuracy(Yhat_train, Variable(data.Ytrain.type(dtype),
            requires_grad=False))
        Ltest, acctest = loss_accuracy(Yhat_test, Variable(data.Ytest.type(dtype),
            requires_grad=False))
        Ygrid, _ = forward(params, Variable(data.Xgrid.type(dtype), requires_grad=False))

        title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain, acctest, Ltest)
        print(title)
        data.plot_data_with_grid(Ygrid, title)
        data.plot_loss(Ltrain, Ltest, acctrain, acctest)

    input("done")


