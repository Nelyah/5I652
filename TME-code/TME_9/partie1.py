import pickle, PIL
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os, torchvision
torchvision.models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch"


vgg16 = torchvision.models.vgg16(pretrained=True)
imagenet_classes = pickle.load(open("imagenet_classes.pkl", "rb"))
#print(imagenet_classes)

# chargement des classes
img = PIL.Image.open("cat4.jpg")
img = img.resize((224, 224), PIL.Image.BILINEAR)
img = np.array(img, dtype=np.float32)
img = img.transpose((2, 0, 1))
img /= 255
mu = np.array([0.485 , 0.456 , 0.406])
sig = np.array([0.229, 0.224, 0.225])
for i in range(3):
   img[i] = (img[i] - mu[i]) / sig[i]
#mu = np.broadcast_to(mu, shape=(3,224,224))
#sig = np.broadcast_to(sig, shape=(3,224,224))

# TODO preprocess image
#img = (img - mu) / sig
img = np.expand_dims(img, 0)

# transformer en batch contenant une image
x = Variable(torch.Tensor(img))
y = vgg16.forward(x)
soft = nn.Softmax()
y = soft(y)

# TODO calcul forward
y = y.data.numpy()
#print(imagenet_classes)
print(y)
print(y.argmax())
print(imagenet_classes[y.argmax()])
# transformation en array numpy
# TODO récupérer la classe prédite et son score de confiance
