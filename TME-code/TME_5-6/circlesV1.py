import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tme6 import CirclesData

def init_params(nx, nh, ny):
    params = {}
    mean = 0
    std = 0.3

    # TODO remplir avec les paramètres Wh, Wy, bh, by
    params["Wh"] = torch.randn(nh, nx) * 0.3
    params["bh"] = torch.randn(nh, 1) * 0.3
    params["Wy"] = torch.randn(ny, nh) * 0.3
    params["by"] = torch.randn(ny, 1) * 0.3

    return params

def forward(params, X):
    outputs = {}

    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    outputs["X"] = X
    outputs["htilde"] = X.mm(params["Wh"].t()) + params["bh"].t().expand(X.shape[0], nh)
    outputs["h"] = torch.tanh(outputs["htilde"])

    outputs["ytilde"] = outputs["h"].mm(params["Wy"].t()) + params["by"].t().expand(X.shape[0], ny)
    # print(torch.nn.Softmax(torch.autograd.Variable(outputs["ytilde"])))
    # outputs["yhat"] = torch.nn.Softmax(torch.autograd.Variable(outputs["ytilde"]))
    outputs["yhat"] = torch.exp(outputs["ytilde"])
    yhat_den = torch.sum(outputs["yhat"], 1, keepdim = True)
    outputs["yhat"] = outputs["yhat"] / yhat_den.expand_as(outputs["yhat"])


    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = 0
    acc = 0

    L = -torch.mean(torch.sum(Y * (torch.log(Yhat)), 1), 0)
    _, indsY = torch.max(Y, 1)
    _, indsYhat = torch.max(Yhat, 1)
    acc = torch.sum(indsY == indsYhat) * 100 / indsY.shape[0]


    return L, acc

def backward(params, outputs, Y):
    grads = {}

    # TODO remplir avec les paramètres Wy, Wh, by, bh
    # grads["Wy"] = ...
    grads["ytilde"] = outputs["yhat"] - Y
    grads["Wy"]     = grads["ytilde"].t().mm(outputs["h"])
    grads["by"]     = torch.sum(grads["ytilde"], 0, keepdim = True).t()
    grads["htilde"] = grads["ytilde"].mm(params["Wy"]) * (1 - outputs["h"]**2)
    grads["Wh"]     = grads["htilde"].t().mm(outputs["X"])
    grads["bh"]     = torch.sum(grads["htilde"], 0, keepdim = True).t()

    return grads

def sgd(params, grads, eta):
    # TODO mettre à jour le contenu de params
    for param_name in ["Wy", "by", "Wh", "bh"]:
        params[param_name] = params[param_name] - eta * grads[param_name]

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
    Yhat, outs = forward(params, data.Xtrain)
    L, _ = loss_accuracy(Yhat, Y)
    grads = backward(params, outs, Y)
    params = sgd(params, grads, eta)

    print(params)
    # TODO apprentissage
    for i in range(Nepoch):
        start = 0
        end = 0
        for j in range(1, int(N/Nbatch)):
            end = j * Nbatch
            # print(j, start, end)
            Xbatch = data.Xtrain[start:end]
            Ybatch = data.Ytrain[start:end]
            start = end

            Yhat, outs = forward(params, Xbatch)
            L, _ = loss_accuracy(Yhat, Ybatch)
            grads = backward(params, outs, Ybatch)
            params = sgd(params, grads, eta)
    print(params)

    # attendre un appui sur une touche pour garder les figures
    input("done")
