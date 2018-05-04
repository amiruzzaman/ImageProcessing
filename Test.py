# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:00:41 2017

@author: mamiruzz
"""

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
    # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion


TRAIN_DIR = 'C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\HouseData\\train'
TEST_DIR = 'C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\HouseData\\test'
TRAIN_DATA_FILE = 'train_data.npy'
TEST_DATA_FILE = 'test_data.npy'
IMG_SIZE = 64
EPOCH = 10
LR = 1e-3

MODEL_NAME = 'AmirNet-{}-{}.model'.format(LR, 'HouseCNN') # just so we remember which saved model is which, sizes must match

def label_img(img):
    word_label = img.split('.')[-3]
  
    if word_label == 'bad': return [1, 0]
    elif word_label == 'well': return [0, 1]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save(TRAIN_DATA_FILE, training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save(TEST_DATA_FILE, testing_data)
    return testing_data

import os.path

if os.path.isfile(TRAIN_DATA_FILE):
    train_data = np.load(TRAIN_DATA_FILE)
else:
    train_data = create_train_data()
    
if os.path.isfile(TEST_DATA_FILE):
    testing_data = np.load(TEST_DATA_FILE)
else:
    testing_data = process_test_data()
    



#define neural network
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input_image') 
#input class
y_ = tf.placeholder(tf.float32, shape=[None, 2], name='input_class')


input_layer = x
#convolutional layer 1 --convolution+RELU activation
conv_layer1 = conv_2d(input_layer, nb_filter=32, filter_size=4, strides=[1, 1, 1, 1],
                                          padding='same', activation='relu', regularizer="L2", name='conv_layer_1')

#2x2 max pooling layer
out_layer1 = max_pool_2d(conv_layer1, 2)


#second convolutional layer 
conv_layer2 = conv_2d(out_layer1, nb_filter=64, filter_size=5, strides=[1, 1, 1, 1],
                                          padding='same', activation='relu', regularizer="L2", name='conv_layer_2')
out_layer2 = max_pool_2d(conv_layer2, 2)
# third convolutional layer
conv_layer3 = conv_2d(out_layer2, nb_filter=32, filter_size=5, strides=[1, 1, 1, 1],
                                          padding='same', activation='relu', regularizer="L2", name='conv_layer_2')
out_layer3 = max_pool_2d(conv_layer3, 2)

#fully connected layer1
fcl = fully_connected(out_layer3, 1024, activation='relu', name='FCL-1')
fcl_dropout_1 = dropout(fcl, 0.8)
#fully connected layer2
fc2 = fully_connected(fcl_dropout_1, 1024, activation='relu', name='FCL-2')

#dropout
fcl_dropout_2 = dropout(fc2, 0.8)

NumberOfClasses = 2

y_predicted = fully_connected(fcl_dropout_2, NumberOfClasses, activation='softmax', name='output')

convnet = regression(y_predicted, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

##loss function
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_predicted + np.exp(-10)), reduction_indices=[1]))
##optimiser -
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
##calculating accuracy of our model 
#correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# model
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

#to separate my features and labels
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

#fit for 1 epochs
model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCH, validation_set=({'input': test_x}, {'targets': test_y}), 
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#save model
model.save(MODEL_NAME)



import matplotlib.pyplot as plt

# if you need to create the data:
#test_data = process_test_data()
# if you already have some saved:
test_data = np.load(TEST_DATA_FILE)

fig = plt.figure()



for num, data in enumerate(test_data[:4]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(2, 2, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label = 'Good'
    else: str_label = 'Bad'
        
    y.imshow(orig, cmap='gray')
    #y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

with open('submission_file.csv', 'w') as f:
    f.write('id,label\n')
            
with open('submission_file.csv', 'a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num, model_out[1]))

print ("Classification complete!")

print("working")