import sys
import os

import time

import numpy as np
import eagerpy as ep
import cv2
import random

import sklearn
from sklearn.metrics import confusion_matrix
from art.defences.preprocessor import FeatureSqueezing
from scipy import ndimage

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras

import utils # we need this

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

# Directory
dir_path = os.path.dirname(os.path.realpath(__file__))



train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data()
gauss, snp, poisson, speckle, none = 0,0,0,0,0
counter = 0
print(len(val_x))
for i in val_x:
    # cv2.imshow()
    counter += 1
    x = random.randint(1,8)
    if x==1:
        ret = noisy("gauss",i)
        gauss += 1
    elif x==2:
        ret = noisy("s&p",i)
        snp += 1
    elif x==3:
        ret = noisy("poisson",i)
        poisson += 1
    elif x==4:
        ret = noisy("speckle",i)
        speckle += 1
    else: 
        ret = i
        none += 1
    if x<=4:
        subfolder = "generatedImages/noisy"
    else:
        subfolder = "generatedImages/nonNoisy"
    subfolder_path = os.path.join(dir_path, subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    img_path = os.path.join(subfolder_path, f"{counter}.jpg")
    cv2.imwrite(img_path, i*255)
print("gauss",gauss)
print("snp",snp)
print("poisson",poisson)
print("speckle",speckle)
print("nonNoisy",none)



