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

models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch"
PRINT_INTERVAL = 20
CUDA = False


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


def extract_features(data, model):
    t_X, t_y = [], []
    print(model)
    for i, (input, target) in enumerate(data):
        input = Variable(input)
        target = Variable(target)
        if i % PRINT_INTERVAL == 0:
            print('Batch {0:03d}/{1:03d}'.format(i, len(data)))
        if CUDA:
            input = input.cuda()
        # TODO Feature extraction a faire
        #print(input.data.shape)
        t_X.append(model.forward(input).data.cpu().numpy())
        t_y.append(target)

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
            self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])

        def forward(self, x):
            print(x.size())
            x = self.features(x)
            x = x.view(x.size(0), -1)
            print(x.size())
            x = self.classifier(x)
            return x

    print('Instanciation de VGG16relu7')
    model = VGG16relu7() # TODO À remplacer pour feature extraction
    model.eval()
    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(params.batch_size, params.path)
    #train, test = Variable(train), Variable(test)

    # Extraction des features
    print('Feature extraction')
    X_train, y_train = extract_features(train, model)
    X_test, y_test = extract_features(test, model)


    # TODO Apprentissage et évaluation des SVM à faire
    print('Apprentissage des SVM')
    svm = LinearSVC(C=1.0)
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
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
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
