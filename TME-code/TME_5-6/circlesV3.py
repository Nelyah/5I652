import math
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tme6 import CirclesData

def init_model(nx, nh, ny, eta):
    model = torch.nn.Sequential(
        torch.nn.Linear(nx, nh),
        torch.nn.Tanh(),
        #torch.nn.Linear(ny, nh),
        torch.nn.Linear(nh, ny),
        torch.nn.Softmax()
    )
    loss = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=eta)

    return model, loss, optim


def loss_accuracy(Yhat, Y, loss_fun):
    L = 0
    acc = 0

    #L = -torch.mean(torch.sum(Y * (torch.log(Yhat)), 1), 0)
    L = loss(Yhat, Ybatch)
    _, indsY = torch.max(Y, 1)
    _, indsYhat = torch.max(Yhat, 1)
    acc = torch.sum(indsY == indsYhat) * 100 / indsY.size(0)


    return L, acc

def sgd(model, eta):
    for param in model.parameters():
        param.data -= eta * param.grad.data
    model.zero_grad()



if __name__ == '__main__':

    # init
    Nepoch = 500
    data = CirclesData()
    data.plot_data()
    Y = data.Ytrain
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    # Premiers tests, code à modifier
    model, loss, optim = init_model(nx, nh, ny, eta)

    # TODO apprentissage
    for i in range(Nepoch):
        start = 0
        end = 0
        for j in range(1, int(N/Nbatch)):
            end = j * Nbatch
            Xbatch = Variable(data.Xtrain[start:end], requires_grad=False)
            Ybatch = Variable(data.Ytrain[start:end], requires_grad=False)
            start = end

            Yhat = model(Xbatch)
            L, _ = loss_accuracy(Yhat, Ybatch, loss)
            optim.zero_grad()
            L.backward()
            optim.step()
        # title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain, acctest, Ltest)
        # print(title)
        # data.plot_data_with_grid(Ygrid, "test")
        # data.plot_loss(Ltrain, Ltest, acctrain, acctest)

    # attendre un appui sur une touche pour garder les figures
    input("done")
