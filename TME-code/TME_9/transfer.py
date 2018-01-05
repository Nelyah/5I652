# -*- coding: utf-8 -*-

global try_y_train
global try_X_train
global try_X_test
global try_y_test 
import argparse
import os
import time

from PIL import Image

import numpy as np

from sklearn.svm import LinearSVC

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision

models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch"
PRINT_INTERVAL = 20
CUDA = False



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, keep_all=False):
        self.reset()
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.data is not None:
            self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def get_dataset(batch_size, path):
    """
    Cette fonction charge le dataset et effectue des transformations sur chaqu
    """
    train_dataset = datasets.ImageFolder(path+'/train',
        transform=transforms.Compose([ # TODO Pre-traitement a faire
            transforms.Lambda(lambda x: x.resize((224, 224,))),
            transforms.ToTensor(),
            transforms.Normalize((0.495, 0.456, 0.406,), (0.229, 0.224, 0.225,))
        ]))
    val_dataset = datasets.ImageFolder(path+'/test',
        transform=transforms.Compose([ # TODO Pre-traitement a faire
            transforms.Lambda(lambda x: x.resize((224, 224,))),
            transforms.ToTensor(),
            transforms.Normalize((0.495, 0.456, 0.406,), (0.229, 0.224, 0.225,))
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)

    return train_loader, val_loader

def epoch(data, model, criterion, optimizer=None):
    model.eval() if optimizer is None else model.train()

    t_X, t_y = [], []
    print(model)
    for i, (input, target) in enumerate(data):
        input = Variable(input)
        target = Variable(target)

        if CUDA:
            input = input.cuda()
            target = target.cuda()

        output = model.forward(input)
        loss = criterion(output, target)

        # backward si on est en "train"
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        if i % PRINT_INTERVAL == 0:
            print('Batch {0:03d}/{1:03d}'.format(i, len(data)))
        # TODO Feature extraction a faire
        #print(input.data.shape)
        t_X.append(model.forward(input).data.cpu().numpy())
        t_y.append(target)


    print("accuracy: ", accuracy(output.data, target.data, topk=(1, 5)))
    # backward si on est en "train"
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return t_X, t_y


def main(params):
    print('Instanciation de VGG16')
    vgg16 = models.vgg16(pretrained=True)

    class VGG16relu7(nn.Module):
        def __init__(self):
            super(VGG16relu7, self).__init__()
            # recopier toute la partie convolutionnelle
            self.features = nn.Sequential(
                    *list(vgg16.features.children()))
            # garder une partie du classifieur, -2 pour s’arreter a relu7
            # self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])

            self.classifier = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(),                     
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 10),
                nn.ReLU()
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    print('Instanciation de VGG16relu7')
    model = VGG16relu7() # TODO À remplacer pour feature extraction

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0)#.9)

    # model = torchvision.models.squeezenet1_1(pretrained=True)
    model.eval()
    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(params.batch_size, params.path)
    #train, test = Variable(train), Variable(test)

    # Extraction des features
    print('Feature extraction')
    for i in range(5):
        X_train, y_train = epoch(train, model, criterion, optimizer)
    X_test, y_test = epoch(test, model, criterion)


    # TODO Apprentissage et évaluation des SVM à faire
    print('Apprentissage des SVM')
    svm = LinearSVC(C=1.0)
    print(svm)
    print(type(X_train), type(y_train))
    y_train = [k.data.cpu().numpy() for k in y_train]
    y_test = [k.data.cpu().numpy() for k in y_test]

    try_y_train = y_train
    try_X_train = X_train
    try_X_test = X_test
    try_y_test = y_test


    X_train = np.concatenate(tuple([k for k in X_train]))
    y_train = np.concatenate(tuple([k for k in y_train]))
    X_test = np.concatenate(tuple([k for k in X_test]))
    y_test = np.concatenate(tuple([k for k in y_test]))
    print(X_train, y_train)
    print(X_train[-1].shape, y_train[-1].shape)
    svm.fit(X_train, y_train)
    accuracy = svm.score(X_test, y_test)
    print("accuracy:", accuracy)

if __name__ == '__main__':

    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='15SceneData/', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--cuda', dest='cuda', action='store_false', help='activate GPU acceleration')

    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True
    else:
        CUDA = False
    print(CUDA)

    main(args)

    input("done")
