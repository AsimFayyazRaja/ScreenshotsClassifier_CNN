import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation
import csv
import string
from collections import Counter
from tqdm import tqdm
import collections, re
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
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import os
import time
import imageio
import plotly.plotly as py
import plotly.graph_objs as go

path="/home/asim/Desktop/Screenshots/model.tflearn.meta"

class PlottingCallback(tflearn.callbacks.Callback):
    def __init__(self, model, x,
                 layers_to_observe=(),
                 kernels=10,
                 inputs=1):
        self.model = model
        self.x = x
        self.kernels = kernels
        self.inputs = inputs
        self.observers = [tflearn.DNN(l) for l in layers_to_observe]

    def on_epoch_end(self, training_state):
        outputs = [o.predict(self.x) for o in self.observers]

        for i in range(self.inputs):
            plt.figure(frameon=False)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            ix = 1
            for o in outputs:
                for kernel in range(self.kernels):
                    plt.subplot(len(outputs), self.kernels, ix)
                    plt.imshow(o[i, :, :, kernel])
                    plt.axis('off')
                    ix += 1
            plt.savefig('outputs-for-image:%i-at-epoch:%i.png'
                        % (i, training_state.epoch))


def generate_training_data(folder):
    r=0
    "Gets images for training, adds labels and returns training data"
    print("Getting images for training..")
    training_data = []
    bag=[]
    label=[]
    with tqdm(total=len(glob.glob(folder+"/*.png"))) as pbar:
        for img in glob.glob(folder+"/*.png"):
            temp=[]
            #if r>=10:
            #    break
            if "fb" in img:
                #tr=0
                tr=[1,0,0,0,0,0]
                n= cv2.imread(img)
            elif "yt" in img:
                #tr=1
                tr=[0,1,0,0,0,0]
                n= cv2.imread(img)
            elif "stack" in img:
                #tr=2
                tr=[0,0,1,0,0,0]
                n= cv2.imread(img)
            elif "gmail" in img:
                #tr=3
                tr=[0,0,0,1,0,0]
                n= cv2.imread(img)
            elif "code" in img:
                #tr=4
                tr=[0,0,0,0,1,0]
                n= cv2.imread(img)
            elif "others" in img:
                #tr=4
                tr=[0,0,0,0,0,1]
                n= cv2.imread(img)
            
            else:
                n= cv2.imread(img)
                tr=[0]
            temp.append(n)
            temp.append(tr)
            bag.append(temp)
            pbar.update(1)
            r+=1
    return bag

def remove_files(imgpath):
    #removing all test files after classifying them
    toremove=imgpath
    filelist = glob.glob(toremove+"/*.png")
    print("Removing files from ",imgpath," ...")
    with tqdm(total=len(filelist)) as pbar:
        for f in filelist:
            os.remove(f)
            pbar.update(1)


if os.path.exists(path):
    print("Loading the model..")
    tf.reset_default_graph()
    convnet=input_data(shape=[None,50,50,3],name='input')
    convnet=conv_2d(convnet,32,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    max_0=conv_2d(convnet,64,5,activation='relu')
    max_1=max_pool_2d(max_0,5)
    convnet=conv_2d(max_1,32,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    max_2=fully_connected(convnet,128,activation='relu')
    convnet=dropout(max_2,0.4)
    max_3=fully_connected(convnet,6,activation='softmax')
    convnet=regression(max_3,optimizer='adam',learning_rate=0.005,loss='categorical_crossentropy',name='ScreenshotClassifier')
    model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=3)
    model.load('./model.tflearn')
else:
    bag=generate_training_data("Screenshots")
    random.shuffle(bag)
    i=0
    data=[]
    labels=[]
    for i in range(len(bag)):       #sepearting features and labels
        data.append(bag[i][0])
        labels.append(bag[i][1])
    del bag
    i=0
    X=[]
    print("Resizing images")
    with tqdm(total=len(data)) as p1bar:
        for i in range(len(data)):
            x=np.array(transform.resize(data[i],[50,50,3]),dtype='float32')
            X.append(x)
            p1bar.update(1)
    del data
    data=X

    X_train, X_test, y_train, y_test=cross_validation.train_test_split(data,labels,test_size=0.1)

    tf.reset_default_graph()
    convnet=input_data(shape=[None,50,50,3],name='input')
    convnet=conv_2d(convnet,32,5,activation='relu')
    max_1=max_pool_2d(convnet,5)
    convnet=conv_2d(max_1,64,5,activation='relu')
    convnet=max_pool_2d(convnet,5)

    convnet=conv_2d(convnet,32,5,activation='relu')
    max_0=max_pool_2d(convnet,5)

    convnet=fully_connected(max_0,128,activation='relu')
    convnet=dropout(convnet,0.4)
    convnet=fully_connected(convnet,6,activation='softmax')
    convnet=regression(convnet,optimizer='adam',learning_rate=0.005,loss='categorical_crossentropy',name='ScreenshotClassifier')
    model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=3)
    model.fit(X_train,y_train, n_epoch=20,validation_set=(X_test,y_test), snapshot_step=20,show_metric=True,
    run_id='ScreenshotClassifier',callbacks=[PlottingCallback(model, X_test, (max_0))])
    print("Saving the model")
    model.save('model.tflearn')
    del X_train
    del y_train
    del X_test
    del y_test

#testing here
bag=generate_training_data("test2")
random.shuffle(bag)

i=0
data=[]
labels=[]
print("Getting test data..")
for i in range(len(bag)):       #sepearting features and labels
    data.append(bag[i][0])      #just images for test data, no labels
del bag

i=0
X=[]
print("Resizing images")
with tqdm(total=len(data)) as p1bar:
    for i in range(len(data)):
        #if i>=90:
        #    break
        x=np.array(transform.resize(data[i],[50,50,3]),dtype='float32')
        X.append(x)
        p1bar.update(1)
X_test=X                     #for feeding to NN for predicting label
real_data=copy.deepcopy(X_test)      #for displaying images in testing
j=len(X_test)
m=j
s=0

ot=0
fb=0
st=0
cod=0
gm=0
yt=0
imgpath="/home/asim/Desktop/Screenshots"        #base path of all classes' folders
print("Predicting on test set..")
'''
observed = [max_0,max_1,max_2,max_3]
observers = [tflearn.DNN(v, session=model.session) for v in observed]
outputs = [m.predict(X_test) for m in observers]
print([d.shape for d in outputs])
kernel=1
for i in [5,55,92,149,240]:
    #i+30
    plt.imshow(outputs[0][i, :, :, kernel])
    plt.show()
'''
'''
observed = [max_0,max_1,max_2]
observers = [tflearn.DNN(v, session=model.session) for v in observed]
outputs = [m.predict(X_test) for m in observers]

kernels=10
for i in range(1):
    plt.figure(frameon=False)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ix = 1
    for o in outputs:
        for kernel in range(kernels):
            plt.subplot(len(outputs),kernels, ix)
            plt.imshow(o[0, :, :, kernel])
            plt.axis('off')
            ix += 1
    plt.savefig('outputs-for-image:%i-at-epoch:%i.png'
                % (i, training_state.epoch))
'''
#plt.savefig('output.png')

'''
for o in outputs:
    for kernel in range(kernels):
        plt.imshow(o[i, :, :, kernel])
        plt.axis('off')
        ix += 1
'''


remove_files(imgpath+"/fb")
remove_files(imgpath+"/yt")
remove_files(imgpath+"/gmail")
remove_files(imgpath+"/stack")
remove_files(imgpath+"/code")
remove_files(imgpath+"/others")

with tqdm(total=m) as p1bar:
    while j>=0:
        num_of_pics_in_graph=16
        fig=plt.figure(figsize=(num_of_pics_in_graph,12))
        i=0
        for r in (X_test):
            ts = time.time()
            img_data=r
            y=fig.add_subplot(4,4,i+1)
            orig=img_data
            model_out=model.predict([img_data])
            model_out=model.predict([img_data])[0]
            if np.argmax(model_out) ==0:
                if os.path.exists(imgpath+"/fb"):
                    str_label='FB'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/fb/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                else:
                    os.makedirs(imgpath+"/fb")
                    str_label='FB'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/fb/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                fb+=1
            elif np.argmax(model_out) ==1:
                if os.path.exists(imgpath+"/yt"):
                    str_label='YT'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/yt/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                else:
                    os.makedirs(imgpath+"/yt")
                    str_label='YT'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/yt/".format(ts)+".png"
                    imageio.imwrite(imgpath+p, data[s])
                yt+=1
            elif np.argmax(model_out) ==2:
                if os.path.exists(imgpath+"/stack"):
                    str_label='Stack'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/stack/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                else:
                    os.makedirs(imgpath+"/stack")
                    str_label='Stack'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/stack/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                st+=1
            elif np.argmax(model_out) ==3:
                if os.path.exists(imgpath+"/gmail"):
                    str_label='Gmail'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/gmail/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                else:
                    os.makedirs(imgpath+"/gmail")
                    str_label='Gmail'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/gmail/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                gm+=1
            elif np.argmax(model_out) ==4:
                if os.path.exists(imgpath+"/code"):
                    str_label='Code'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/code/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                else:
                    os.makedirs(imgpath+"/gmail")
                    str_label='Gmail'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/gmail/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                cod+=1
            else:
                if os.path.exists(imgpath+"/others"):
                    str_label='Others'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/others/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                else:
                    os.makedirs(imgpath+"/others")
                    str_label='Others'
                    ts=str(ts)
                    ts=ts.replace('.','_')
                    p="/others/"+ts+".png"
                    imageio.imwrite(imgpath+p, data[s])
                ot+=1
            y.imshow(data[s])          #showing real image on figure
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
            i+=1
            j-=1
            s+=1
            p1bar.update(1)
            if(j<=0):
                j=-2
                break
            if(i>=15):        #cause plt figure size
                break
        if(j>=0):
            X_test=copy.deepcopy(X_test[15:])
        #plt.show()

remove_files(imgpath+"/test")


labels = 'FB', 'YT', 'Code', 'Stack','Gmail','Others'
sizes = [fb,yt,cod,st,gm,ot]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','brown','red']
explode = (0,0,0,0,0,0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
#plt.show()

