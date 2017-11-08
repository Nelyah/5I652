import math
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tme6 import CirclesData


def init_params(nx, nh, ny):
    params = {}
    mean = 0
    std = 0.3

    # TODO remplir avec les paramètres Wh, Wy, bh, by
    params["Wh"] = Variable(torch.randn(nh, nx), requires_grad=True) * 0.3
    params["bh"] = Variable(torch.zeros(nh, 1), requires_grad=True) * 0.3
    params["Wy"] = Variable(torch.randn(ny, nh), requires_grad=True) * 0.3
    params["by"] = Variable(torch.zeros(ny, 1), requires_grad=True) * 0.3

    return params

def forward(params, X):
    outputs = {}

    X_size = X.size(0)

    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    outputs["X"] = X
    outputs["htilde"] = X.mm(params["Wh"].t()) + params["bh"].t().expand(X_size, nh)
    outputs["h"] = torch.tanh(outputs["htilde"])

    outputs["ytilde"] = outputs["h"].mm(params["Wy"].t()) + params["by"].t().expand(X_size, ny)
    # print(torch.nn.Softmax(torch.autograd.Variable(outputs["ytilde"])))
    # outputs["yhat"] = torch.nn.Softmax(torch.autograd.Variable(outputs["ytilde"]))
    outputs["yhat"] = torch.exp(outputs["ytilde"])
    yhat_den = torch.sum(outputs["yhat"], 1, keepdim = True)
    outputs["yhat"] = outputs["yhat"] / yhat_den.expand_as(outputs["yhat"])

    print("hat",outputs["yhat"].requires_grad)
    print("tilde",outputs["ytilde"].requires_grad)

    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = 0
    acc = 0

    L = -torch.mean(torch.sum(Y * (torch.log(Yhat)), 1), 0)
    _, indsY = torch.max(Y, 1)
    _, indsYhat = torch.max(Yhat, 1)
    acc = torch.sum(indsY == indsYhat) * 100 / indsY.size(0)


    return L, acc

def sgd(params, eta):
    print("sgd")
    # TODO mettre à jour le contenu de params
    for param_name in ["Wy", "by", "Wh", "bh"]:
        print(params[param_name].requires_grad)
        params[param_name].data -= eta * params[param_name].grad.data
        params[param_name].grad.data.zero_()
    exit()

    return params



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
    params = init_params(nx, nh, ny)
    print("par2",params["Wy"].grad_fn)

    # TODO apprentissage
    for i in range(Nepoch):
        start = 0
        end = 0
        for j in range(1, int(N/Nbatch)):
            end = j * Nbatch
            # print(j, start, end)
            Xbatch = Variable(data.Xtrain[start:end], requires_grad=False)
            Ybatch = Variable(data.Ytrain[start:end], requires_grad=False)
            start = end

            Yhat, outs = forward(params, Xbatch)
            print("type",type(Yhat), type(Ybatch))
            L, _ = loss_accuracy(Yhat, Ybatch)
            print(type(L))
            print("L", L.creator)
            print("par1",params["by"].grad_fn)
            L.backward()

            print("par2",params["by"])
            print("par3",params["by"].requires_grad)
            print("par4",params["by"].grad_fn)
            params = sgd(params, eta)
    print(params)

    # attendre un appui sur une touche pour garder les figures
    input("done")
