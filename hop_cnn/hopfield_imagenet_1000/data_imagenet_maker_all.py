# import library 

import os
import re
import cv2
import numpy as np
import glob
from keras import utils
from keras.preprocessing.image import  img_to_array, load_img
from PIL import ImageFile
from PIL import Image

# fix bug
ImageFile.LOAD_TRUNCATED_IMAGES = True

# make image
class DataImageMaker():

    def __init__(self, data_dir, num_image, img_size=250, l=0):
        # set parameter 
        self.data_dir = data_dir
        self.num_image = num_image
        self.l = l
        self.lr = l
        self.one_step = 100
        self.img_size = img_size

    def addGaussianNoise(self, src, var, sigma):
        # set parameter
        row,col,ch= src.shape
        mean = 0
        var = var
        sigma = sigma
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = src + gauss
        return noisy

    def gen_ori(self):
        # load original image 

        l = self.l
        self.random_counter()
        img_rows, img_cols = self.img_size, self.img_size
        data_dir = self.data_dir
        cls_name = [i for i in os.listdir(data_dir) if os.path.isdir(data_dir +i)]
        num_classes = len(cls_name) 
        self.num_classes = num_classes
        num = self.num_image
                
        imagenet = []
        imagenet_class = []

        for cl_num, cls in enumerate(cls_name):
            pct_list = glob.glob(os.path.join(data_dir, cls, "*"))
            
            if len(pct_list) == 0: continue
            for i in range(num):
                np.random.seed(i+l)
                idx = np.random.randint(low=0,high=len(pct_list))
                img = img_to_array(load_img(pct_list[idx], target_size=(img_rows, img_cols)))
                imagenet.append(img)
                imagenet_class.append(cl_num)

        imagenet = np.asarray(imagenet)
        imagenet_class = np.asarray(imagenet_class)

        x_train = imagenet
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = imagenet_class
        y_train = utils.to_categorical(y_train, num_classes)
        self.x_train = x_train
        self.y_train = y_train

        return x_train, y_train

    def gen_noize(self, var = 0.1 ,sigma = 30):
        # load image and generate noise
        
        self.var = var
        self.sigma = sigma
        l = self.lr
        self.random_counter(noize=True)
        img_rows, img_cols = self.img_size, self.img_size
        data_dir = self.data_dir
        cls_name = [i for i in os.listdir(data_dir) if os.path.isdir(data_dir +i)]
        num_classes = len(cls_name) 
        self.num_classes = num_classes
        num = self.num_image
                
        imagenet = []
        imagenet_class = []

        for cl_num, cls in enumerate(cls_name):
            pct_list = glob.glob(os.path.join(data_dir, cls, "*"))
           
            if len(pct_list) == 0: continue
            for i in range(num):
                np.random.seed(i+l)
                idx = np.random.randint(low=0,high=len(pct_list))
                img = img_to_array(load_img(pct_list[idx], target_size=(img_rows, img_cols)))
                imagenet.append(self.addGaussianNoise(img, var, sigma))
                imagenet_class.append(cl_num)

        imagenet = np.asarray(imagenet)
        imagenet_class = np.asarray(imagenet_class)

        x_train = imagenet
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = imagenet_class
        y_train = utils.to_categorical(y_train, num_classes)

        self.noize_x = x_train
        self.noize_y = y_train

        return x_train, y_train

    def random_counter(self, noize = False):
        if noize: self.lr = self.lr +self.one_step
        else : self.l = self.l +self.one_step

    def save_image(self,noize = False):
        if noize: train_num = len(self.noize_x)
        else: train_num = len(self.x_train)
        num = self.num_image
        num_classes = self.num_classes

        save_dir = 'data'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if noize :
            name = "imagenet_{cl}_gaussian_{var}_{sig}_sim_{n}_{irow}_rand_{l}".format(cl = num_classes, 
            var=str(self.var).replace(".", ""), sig = self.sigma, n = num, irow =self.img_size, l = (self.l-self.one_step))  
            name_x = name + "_x"
            name_y = name + "_y"   
            np.save(os.path.join(save_dir, name_x), self.noize_x)
            np.save(os.path.join(save_dir, name_y), self.noize_y)      
        else: 
            name = "imagenet_" + str(num_classes)+ "_sim_" + str(num) + "_" + str(self.img_size) + "_rand_" + str(self.l-self.one_step)
            name_x = name + "_x"
            name_y = name + "_y"
            np.save(os.path.join(save_dir, name_x), self.x_train)
            np.save(os.path.join(save_dir, name_y), self.y_train)

