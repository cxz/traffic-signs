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
    

def read_train():
    training_file = 'data/train.p'
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    X_train, y_train = train['features'], train['labels']
    return X_train, y_train

def read_test():
    testing_file = 'data/test.p'
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    
    X_test, y_test = test['features'], test['labels']
    return X_test, y_test
    
def build_train():
    X_train, y_train = read_train()
    x_train = np.array([process_image(x) for x in X_train], dtype=np.float32)
    y_train = y_train.astype(np.uint8)
    return train_test_split(x_train, y_train, test_size=0.20, stratify = y_train ) 
        
def build_test():
    X_test, y_test = read_test()
    x_test = np.array([process_image(x) for x in X_test], dtype=np.float32)
    y_test = y_test.astype(np.uint8)
    return x_test, y_test
    
def process_image(img):    
    for i in range(3):
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    img = img.astype(np.float32)
    img /= 255
    return img    
    
def jitter(img):
    degree_range = 15
    gamma_range = 0.2
    gaussian = 0.2
    
    #rotation
    degrees = np.random.randint(-degree_range, degree_range)
    img = rotate(img, degrees)    
    #gamma
    img = adjust_gamma(img, np.random.uniform(1 - gamma_range, 1 + gamma_range))
    if np.random.rand() < gaussian:
        img = random_noise(img, mode='localvar')
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

def next_batch(x_data, y_data, batch_size, step):
    batch_start = step*batch_size
    batch_x = x_data[batch_start:batch_start + batch_size]
    batch_y = y_data[batch_start:batch_start + batch_size]    
    return batch_x, batch_y

def train_generator(x_data, y_data, batch_size):
    while True:
        for step in range(len(x_data)//batch_size):
            batch_start = step*batch_size
            batch_x = np.array([jitter(img) for img in x_data[batch_start:batch_start + batch_size]])
            batch_y = y_data[batch_start:batch_start + batch_size]    
            yield batch_x, batch_y
        
def evaluate(sess, loss_op, accuracy_op, x, y, keep_prob, x_data, y_data):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    steps_per_epoch = len(x_data) // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        batch_x, batch_y = next_batch(x_data, y_data, BATCH_SIZE, step)      
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})        
        total_acc += (acc * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_loss/num_examples, total_acc/num_examples

def train(sess,
          input_tensors, 
          loss_op, accuracy_op,
          data_generator, evaluation, 
          learning_rate, steps_per_epoch, augmentation_factor,
          model_name):
    
    x, y, keep_prob = input_tensors

    # dynamic learning rate
    global_step = tf.Variable(0, trainable=False)
    boundaries = [10, 20, 30] #epochs
    values = [learning_rate, 0.5*learning_rate, 0.1*learning_rate]
    lr = tf.train.piecewise_constant(global_step, boundaries, values)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_op)
    
    saver = tf.train.Saver(max_to_keep=5)
    
    sess.run(tf.global_variables_initializer())
    
    #ckpt = tf.train.get_checkpoint_state('.')
    #if ckpt and ckpt.model_checkpoint_path:
    #    print(ckpt.model_checkpoint_path)
    #    saver.restore(sess, ckpt.model_checkpoint_path)
    #    print("checkpoint restored.")
            
    for i in range(EPOCHS):        
        for step in range(steps_per_epoch):
            batch_x, batch_y = next(data_generator)
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        val_loss, val_acc = evaluation(sess, loss_op, accuracy_op, x, y, keep_prob)            
        print("epoch {}, loss={:.4f}, acc={:.4f}".format(i+1, val_loss, val_acc))
        global_step.assign_add(1)
        saver.save(sess, model_name, global_step=i)
        

EPOCHS = 30
BATCH_SIZE = 200
LEARNING_RATE = 1e-3
NUM_CLASSES = 43
CHANNELS = 3
AUGMENTATION_FACTOR = 5

if __name__ == '__main__':
    X_visible, X_val, y_visible, y_val = build_train()
    
    generator = train_generator(X_visible, y_visible, BATCH_SIZE)
    steps_per_epoch = (AUGMENTATION_FACTOR * len(X_visible)) // BATCH_SIZE
    evaluation = partial(evaluate, x_data=X_val, y_data=y_val)
    
    tensor_x = tf.placeholder(tf.float32, (None, 32, 32, CHANNELS))
    tensor_y = tf.placeholder(tf.int32, (None))
    tensor_keep_prob = tf.placeholder(tf.float32)
    
    y_ = tf.one_hot(tensor_y, NUM_CLASSES)
    logits, weights = model(tensor_x, tensor_keep_prob, CHANNELS, NUM_CLASSES)
    xentropy =  tf.nn.softmax_cross_entropy_with_logits(logits, y_)
    l2_loss = sum([tf.nn.l2_loss(w) for w in weights])
    loss_op = tf.reduce_mean(xentropy) + 1e-5 * l2_loss
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
        train(sess,
              (tensor_x, tensor_y, tensor_keep_prob), loss_op, accuracy_op,
              generator, evaluation, 
              LEARNING_RATE, steps_per_epoch, AUGMENTATION_FACTOR,
              'model1')
    
        print('running test')
        X_test, y_test = build_test()
        test_loss, test_acc = evaluate(sess, 
                                       loss_op, accuracy_op, 
                                       tensor_x, tensor_y, tensor_keep_prob, 
                                       X_test, y_test)
        print('test loss={:.4f}, acc={:.4f}'.format(test_loss, test_acc))
        
"""
epoch 1, loss=0.2066, acc=0.9738
epoch 2, loss=0.1520, acc=0.9869
epoch 3, loss=0.1291, acc=0.9929
epoch 4, loss=0.1200, acc=0.9940
epoch 5, loss=0.1120, acc=0.9950
epoch 6, loss=0.1109, acc=0.9938
epoch 7, loss=0.1016, acc=0.9963
epoch 8, loss=0.1016, acc=0.9946
epoch 9, loss=0.0936, acc=0.9962
epoch 10, loss=0.0892, acc=0.9960
running test
test loss=0.2256, acc=0.9637

epoch 1, loss=0.2788, acc=0.9453
epoch 2, loss=0.1731, acc=0.9769
epoch 3, loss=0.1358, acc=0.9891
epoch 4, loss=0.1231, acc=0.9905
epoch 5, loss=0.1099, acc=0.9942
epoch 6, loss=0.1083, acc=0.9935
epoch 7, loss=0.1067, acc=0.9937
epoch 8, loss=0.0984, acc=0.9945
epoch 9, loss=0.0929, acc=0.9962
epoch 10, loss=0.0920, acc=0.9953
epoch 11, loss=0.0857, acc=0.9959
epoch 12, loss=0.0825, acc=0.9972
epoch 13, loss=0.0860, acc=0.9958
epoch 14, loss=0.0798, acc=0.9964
epoch 15, loss=0.0827, acc=0.9963
epoch 16, loss=0.0831, acc=0.9964
epoch 17, loss=0.0768, acc=0.9965
epoch 18, loss=0.0742, acc=0.9977
epoch 19, loss=0.0737, acc=0.9973
epoch 20, loss=0.0749, acc=0.9972
epoch 21, loss=0.0751, acc=0.9973
epoch 22, loss=0.0716, acc=0.9976
epoch 23, loss=0.0766, acc=0.9965
epoch 24, loss=0.0703, acc=0.9977
epoch 25, loss=0.0707, acc=0.9976
epoch 26, loss=0.0720, acc=0.9972
epoch 27, loss=0.0722, acc=0.9971
epoch 28, loss=0.0723, acc=0.9968



epoch 1, loss=0.2788, acc=0.9453
epoch 2, loss=0.1731, acc=0.9769
epoch 3, loss=0.1358, acc=0.9891
epoch 4, loss=0.1231, acc=0.9905
epoch 5, loss=0.1099, acc=0.9942
epoch 6, loss=0.1083, acc=0.9935
epoch 7, loss=0.1067, acc=0.9937
epoch 8, loss=0.0984, acc=0.9945
epoch 9, loss=0.0929, acc=0.9962
epoch 10, loss=0.0920, acc=0.9953
epoch 11, loss=0.0857, acc=0.9959
epoch 12, loss=0.0825, acc=0.9972
epoch 13, loss=0.0860, acc=0.9958
epoch 14, loss=0.0798, acc=0.9964
epoch 15, loss=0.0827, acc=0.9963
epoch 16, loss=0.0831, acc=0.9964
epoch 17, loss=0.0768, acc=0.9965
epoch 18, loss=0.0742, acc=0.9977
epoch 19, loss=0.0737, acc=0.9973
epoch 20, loss=0.0749, acc=0.9972
epoch 21, loss=0.0751, acc=0.9973
epoch 22, loss=0.0716, acc=0.9976
epoch 23, loss=0.0766, acc=0.9965
epoch 24, loss=0.0703, acc=0.9977
epoch 25, loss=0.0707, acc=0.9976
epoch 26, loss=0.0720, acc=0.9972
epoch 27, loss=0.0722, acc=0.9971
epoch 28, loss=0.0723, acc=0.9968
epoch 29, loss=0.0729, acc=0.9979
epoch 30, loss=0.0741, acc=0.9962
running test
test loss=0.2514, acc=0.9624
"""        