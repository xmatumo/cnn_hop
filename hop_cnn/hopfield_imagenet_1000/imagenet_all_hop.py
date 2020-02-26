import sys
import glob
import keras
import numpy as np
import random as rnd
import tensorflow as tf
from numpy.random import *
import time
import gc
import os
import copy
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential, Model, load_model
from keras.backend import tensorflow_backend

from xnornet.binary_ops import binary_tanh 
# from xnornet.xnor_layers import XnorDense, XnorConv2D
from xnornet.binary_layers import BinaryDense, BinaryConv2D, Clip

from hopfield import Network, covariance_update, extended_storkey_update,hebbian_update
from hopfield_imagenet_1000.hopfield_func import cos_sim, train, iterate_network

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

rnd.seed(a = 1234)

class Hopfield_Restore():

    def __init__(self, face_image, noize_image, noize_image_y, data_dir, model_name):
        self.face_image = face_image
        self.noize_image = noize_image
        self.noize_image_cls = noize_image_y
        self.data_dir = data_dir
        self.mname = model_name


    def calc(self):
        face_images = self.face_image
        noize_images = self.noize_image
        noize_images_cls = self.noize_image_cls

        mid_layer_dim = 10000
        
        save_dir = os.path.join('saved_models')

        model_name = self.mname

        model = load_model(os.path.join(save_dir, model_name),
                            custom_objects={'BinaryDense':BinaryDense, 'BinaryConv2D':BinaryConv2D, 'Clip':Clip, 'binary_tanh':binary_tanh})

        size = face_images.shape[0]
        noize_face = noize_images

        null_sub_train = np.zeros((face_images.shape[0], mid_layer_dim))
        null_sub_face = np.zeros((face_images.shape[0], mid_layer_dim))

        null_sub_noize = np.zeros((noize_face.shape[0], mid_layer_dim))
        null_main_train = np.zeros(face_images.shape)

        ### encode model 
        hidden_layer_name = "mid2"
        start = time.time()
        encode_model = Model(inputs=model.input, outputs=model.get_layer(hidden_layer_name).output)

        encoded_face = encode_model.predict([face_images, null_sub_face])
        encoded_noize = encode_model.predict([noize_images, null_sub_noize])

        null_output = encode_model.predict([null_main_train[0:1], null_sub_train[0:1]])

        denoize_size = face_images.shape[0]
        neuron = copy.deepcopy(encoded_face)

        BATCH_SIZE = 1
        num_setp = 100
        mshape = list(face_images.shape)
        mshape[0] = denoize_size
        mshape = tuple(mshape)
        null_main = np.zeros(mshape)

        out_dir = 'pct_output'

        network = Network(mid_layer_dim)
        
        start = time.time()
        with tf.Session() as sess:

            train(sess, network, neuron, mid_layer_dim)
            iter_state = iterate_network(sess, network, neuron, num_setp)
            test_state = iterate_network(sess, network, encoded_noize, num_setp)
                
        state = iter_state[(num_setp-1)]
        t_state = test_state[(num_setp-1)]

        step_image = []
        step_image.append(model.predict([null_main, (neuron - null_output)]))
        for i, step_mid in enumerate(iter_state):
            step1 = model.predict([null_main, (step_mid - null_output)])
            step_image.append(step1)
            
        test_step = []
        test_step.append(model.predict([null_main, (encoded_noize - null_output)]))
        for i, step_mid in enumerate(test_state):
            step1 = model.predict([null_main, (step_mid - null_output)])
            test_step.append(step1)

        iter_cos_distance = []
        _temp = []

        for noi, ori in zip(encoded_face, neuron):
            _temp.append(cos_sim(noi, ori))
            
        iter_cos_distance.append(_temp)

        for i,st in enumerate(iter_state):
            _step = []
            for m, temp in enumerate(st):
                ss = cos_sim( neuron[m], temp)
                _step.append(ss)
            iter_cos_distance.append(_step)


        iter_cos_after_distance = []
        _temp = []
        for noi, ori in zip(encoded_noize, neuron):
            _temp.append(cos_sim(noi, ori))
            
        iter_cos_after_distance.append(_temp)

        for i,st in enumerate(test_state):
            _step_af = []
            for m, temp in enumerate(st):
                ss = cos_sim( neuron[m], temp)
                _step_af.append(ss)
            iter_cos_after_distance.append(_step_af)


        test_step= np.asarray(test_step)
        step_image = np.asarray(step_image)

        data_size = len(face_images)
        indx =[]
        data_dir = self.data_dir
        cls_name = [i for i in os.listdir(data_dir)  if os.path.isdir(data_dir +i) ]
        for i in range(data_size):
            indx.append(sum(abs(test_step[0,i,:] - test_step[99,i,:]))/2)
        indx = np.asarray(indx)
        indx = np.mean(indx)


        flow = []
        for i in range(data_size):
            flow.append([np.argmax(step_image[0,i,:]), np.max(step_image[0,i,:]), np.argmax(step_image[100,i,:]), np.max(step_image[99,i,:])])
        flow = pd.DataFrame(flow)

        flow_test = []
        for i in range(data_size):
            flow_test.append([np.argmax(test_step[0,i,:]), np.max(test_step[0,i,:]), np.argmax(test_step[100,i,:]), np.max(test_step[99,i,:])])
        flow_test = pd.DataFrame(flow_test)

        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        generator = train_datagen.flow_from_directory(
            data_dir, target_size = (250,250), class_mode = 'categorical',
            batch_size = 32, subset = 'training', seed = 7)
        true_sample = []
        for n in cls_name:
            cl_n = generator.class_indices[n]
            for i in range(1):
                true_sample.append(cl_n)


        true_train=true_sample
        true_test=true_sample
        ori_acc = sum(true_train == flow[0])/len(true_train)
        ori_after_acc = sum(true_train == flow[2])/len(true_train)
        noize_acc = sum(true_test == flow_test[0])/len(true_train)
        noize_after_acc = sum(true_test == flow_test[2])/len(true_train)

        indx_array=np.array([ori_acc, ori_after_acc, noize_acc, noize_after_acc])

        self.indx_array = indx_array
        
        return indx_array, np.array(iter_cos_distance), np.array(iter_cos_after_distance), np.array(step_image), np.array(test_step)

    def save_result(self, name):

        np.save(os.path.join("exp_data","hopfield_mobilenetv2_1000_{}".format(name)), self.indx_array)
    