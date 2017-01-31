# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from skimage.util import random_noise
import cv2
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from functools import partial
import skimage
from scipy.misc import imread
import seaborn as sns

#MEAN = (77.501, 77.801, 78.573)

def plot_signs(X_train, y_train):
    sign_names = pd.read_csv('signnames.csv').set_index('ClassId').to_dict()['SignName']
    for klass in (set(y_train)):
        signs = [x for x,y in zip(X_train, y_train) if y==klass]    
        displayed = min(10, len(signs))
        np.random.shuffle(signs)
        plt.figure()
        f = plt.imshow(np.hstack(signs[0:displayed]))
        plt.title("class: {} -- {}".format(klass, sign_names[klass]))
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    

def build_extra():
    X_extra = []
    for filename in ['110', '80', 'a1', 'a2', 'a3', 'a4', 'b1', 'c1', 'c2', 'd1', 'd2', 'e1', 'e2']:
        image = imread('extra/{}.png'.format(filename))
        image = image[:,:,:3] #exclude last channel (transparency)
        image = skimage.transform.resize(image, (32, 32))
        image = cv2.convertScaleAbs(255 * image)
        image = process_image(image)
        X_extra.append(image)
    return np.array(X_extra)
    
def process_image(img):    
    for i in range(3):
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    img = img.astype(np.float32)
    img /= 255
    return img    
    

def conv2d(x, filter_size, input_channels, output_channels, pooling=False):
    filter_shape = [filter_size,filter_size,input_channels,output_channels]
    weights = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=0, stddev=0.1))    
    biases = tf.Variable(tf.constant(0.0, shape=[output_channels]))
    
    out = tf.nn.bias_add(tf.nn.conv2d(x, weights, strides=[1,1,1,1], padding='SAME'), biases)
    if pooling:
        out = tf.nn.max_pool(out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    out = tf.nn.relu(out)
    return out, weights    

def fc(x, input_channels, output_channels, activation=True):
    weights = tf.Variable(tf.truncated_normal(shape=[input_channels, output_channels], mean=0, stddev=0.1))
    biases = tf.Variable(tf.constant(0.0, shape=[output_channels]))
    out = tf.matmul(x, weights) + biases
    if activation:
        out = tf.nn.relu(out)        
    return out, weights
    
def model(x, keep_prob, channels, num_classes):
    c1, c1_weights = conv2d(x, 1, channels, 3)
    c2, c2_weights = conv2d(c1, 3, 3, 16)    
    c3, c3_weights = conv2d(c2, 5, 16, 16, pooling=True)    
    c3 = tf.nn.dropout(c3, keep_prob)
    flat3 = flatten(c3)
    
    c4, c4_weights = conv2d(c3, 5, 16, 32)    
    c5, c5_weights = conv2d(c4, 5, 32, 32)
    c5 = tf.nn.dropout(c5, keep_prob)
    flat5 = flatten(c5)
    
    fc0 = tf.concat(1, [flat3, flat5])
    fc1, fc1_weights = fc(fc0, 12288, 256)
    fc2, fc2_weights = fc(fc1, 256, 256)
    fc3, fc3_weights = fc(fc2, 256, num_classes, activation=False)
    return fc3, [c1_weights, c2_weights, c3_weights, c4_weights, c5_weights, fc1_weights, fc2_weights, fc3_weights]


NUM_CLASSES = 43
CHANNELS = 3

if __name__ == '__main__':
    #X_test, y_test = build_test()
    X_extra = build_extra()
    sign_names = pd.read_csv('signnames.csv').set_index('ClassId').to_dict()['SignName']
    
    tensor_x = tf.placeholder(tf.float32, (None, 32, 32, CHANNELS))
    #tensor_y = tf.placeholder(tf.int32, (None))
    tensor_keep_prob = tf.placeholder(tf.float32)
    
    #y_ = tf.one_hot(tensor_y, NUM_CLASSES)
    logits, weights = model(tensor_x, tensor_keep_prob, CHANNELS, NUM_CLASSES)
    top_5 = tf.nn.top_k(logits, 5)
    
    #xentropy =  tf.nn.softmax_cross_entropy_with_logits(logits, y_)
    #l2_loss = sum([tf.nn.l2_loss(w) for w in weights])
    #loss_op = tf.reduce_mean(xentropy) + 1e-5 * l2_loss
    
    #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    #accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('.')
        
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("checkpoint {} restored.".format(ckpt.model_checkpoint_path))

        print('running test')
        
        preds = sess.run(top_5, feed_dict={tensor_x: X_extra, tensor_keep_prob:1.0})
        for indices, values, img, in zip(preds.indices, preds.values, X_extra):
            labels = [sign_names[i] for i in indices]
            f, ax = plt.subplots(1, 2, figsize=(3,3))
            sns.barplot(x=values, y=labels, ax=ax[0], orient='h', palette='Blues_d')
            ax[1].imshow(img)
            
            
            
        #print('test loss={:.4f}, acc={:.4f}'.format(test_loss, test_acc))
