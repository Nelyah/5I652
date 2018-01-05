import math
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tme6 import *

def init_model(nx, nh, ny, eta):
    model = torch.nn.Sequential(
        torch.nn.Linear(nx, nh),
        torch.nn.ReLU(),
        torch.nn.Linear(nh, ny),
        torch.nn.LogSoftmax(dim=1),
    )
    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)

    return model, loss, optimizer


def loss_accuracy(Yhat, Y, loss_fun):
    size = Y.size(0)
    _, indsY = torch.max(Y, 1)
    _, indsYhat = torch.max(Yhat, 1)
    L = loss_fun(Yhat, indsY)
    acc = torch.sum(indsY == indsYhat).float() / size
    return L, acc

if __name__ == '__main__':

    # init
    Nepoch = 500
    data = MNISTData()
#    data.plot_data()
    N = data.Xtrain.shape[0]
    Ntest = data.Xtest.shape[0]
    Nbatch = 64
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.003

    # Premiers tests, code à modifier
    model, loss, optimizer = init_model(nx, nh, ny, eta)


#    print(list(model.parameters()))

    # TODO apprentissage
    for iteration in range(Nepoch):
        accu = 0
        accL = 0
        start = 0
        end = 0
        for j in range(1, int(N/Nbatch)):
            end = j * Nbatch
            Xbatch = Variable(data.Xtrain[start:end]/255)
            Ybatch = Variable(data.Ytrain[start:end])
            start = end

            Yhat = model.forward(Xbatch)
            L, acc = loss_accuracy(Yhat, Ybatch, loss)
            accu += acc
            accL += L
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
        Ltrain, acctrain = accL.data[0] / int(N/Nbatch), accu.data[0] /int(N/Nbatch)
#        Ltrain, acctrain = loss_accuracy(model.forward(Variable(data.Xtrain/255, requires_grad = False)), Variable(data.Ytrain, requires_grad = False), loss)
#        Ltrain, acctrain = Ltrain.data[0], acctrain.data[0]
        accu = 0
        accL = 0
        start = 0
        end = 0
        for j in range(1, int(Ntest/Nbatch)):
            end = j * Nbatch
            Xbatch = Variable(data.Xtest[start:end]/255)
            Ybatch = Variable(data.Ytest[start:end])
            start = end

            Yhat = model.forward(Xbatch)
            L, acc = loss_accuracy(Yhat, Ybatch, loss)
            accu += acc
            accL += L

        Ltest, acctest = accL.data[0] / int(Ntest/Nbatch), accu.data[0] /int(Ntest/Nbatch)
#        Ltest, acctest = loss_accuracy(model.forward(Variable(data.Xtest/255, requires_grad = False)), Variable(data.Ytest, requires_grad = False), loss)
#        Ltest, acctest = Ltest.data[0], acctest.data[0]
        title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain, acctest, Ltest)
        print(title)
        data.plot_loss(Ltrain, Ltest, acctrain, acctest)
    """

    Ygrid = model(Variable(data.Xgrid)).data.exp()
    data.plot_data_with_grid(Ygrid, title)
    """
    # attendre un appui sur une touche pour garder les figures
    input("done")
