import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation
import csv
import string
from collections import Counter
from tqdm import tqdm
import collections
import re
import random
from random import randint
from sklearn.metrics import average_precision_score
import pandas as pd
from scipy import misc as cv2
import glob
import tensorflow as tf
from PIL import Image
from skimage import transform
import copy
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os
import time

import warnings
warnings.filterwarnings("ignore")


def generate_training_data(folder):
    r = 0
    "Gets images for training, adds labels and returns training data"
    print("Getting images for training..")
    training_data = []
    bag = []
    label = []
    with tqdm(total=len(glob.glob(folder+"/*.png"))) as pbar:
        for img in glob.glob(folder+"/*.png"):
            temp = []
            if "fb" in img:
                #tr=0
                tr = [1, 0, 0, 0, 0, 0]
                n = cv2.imread(img)
            elif "yt" in img:
                #tr=1
                tr = [0, 1, 0, 0, 0, 0]
                n = cv2.imread(img)
            elif "stack" in img:
                #tr=2
                tr = [0, 0, 1, 0, 0, 0]
                n = cv2.imread(img)
            elif "gmail" in img:
                #tr=3
                tr = [0, 0, 0, 1, 0, 0]
                n = cv2.imread(img)
            elif "code" in img:
                #tr=4
                tr = [0, 0, 0, 0, 1, 0]
                n = cv2.imread(img)
            elif "others" in img:
                #tr=4
                tr = [0, 0, 0, 0, 0, 1]
                n = cv2.imread(img)

            else:
                n = cv2.imread(img)
                tr = [0]
            temp.append(n)
            temp.append(tr)
            bag.append(temp)
            pbar.update(1)
            r += 1
    return bag


bag = generate_training_data("/floyd/input/data/resized_data")
random.shuffle(bag)
i = 0
data = []
labels = []
for i in range(len(bag)):  # sepearting features and labels
    data.append(bag[i][0])
    labels.append(bag[i][1])
del bag

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, labels, test_size=0.1)

tf.reset_default_graph()
convnet = input_data(shape=[None, 128, 128, 3], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
max_1 = max_pool_2d(convnet, 5)
convnet = conv_2d(max_1, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
max_0 = max_pool_2d(convnet, 5)

convnet = fully_connected(max_0, 128, activation='relu')
convnet = dropout(convnet, 0.4)
convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.005,
                     loss='categorical_crossentropy', name='ScreenshotClassifier')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=3)
model.fit(X_train, y_train, batch_size=32, n_epoch=20, validation_set=(X_test, y_test), snapshot_step=20, show_metric=True,
          run_id='ScreenshotClassifier')
print("Saving the model")
model.save('model.tflearn')
del X_train
del y_train
del X_test
del y_test