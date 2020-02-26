import sys
import glob
import os
import gc
import copy
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import pandas as pd
from numpy.random import *
from hopfield_imagenet_1000.image_error_remover import Image_Error_Remover
from hopfield_imagenet_1000.data_imagenet_maker_all import DataImageMaker
from hopfield_imagenet_1000.imagenet_all_hop import Hopfield_Restore

args = sys.argv
l = int(args[1])

epoch = int(args[2])
data_dir = "/aist/data/imagenet_all/ILSVRC2011_images_train/"
model_name = 'keras_mobilenetv2_freez_1000_epoch_{}.h5'.format(epoch) 
# model_name = 'keras_mobilenetv2_freez_1000_epoch30_class_args_10_10000_0.0003.h5' 
# model_name = 'keras_mobilenetv2_freez_1000_con_class_args_10_10000_0.0003.h5' 

save_dir = 'result'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
num_image = 1

print("Remove Gabeage data ")
# Image_Error_Remover().remove(data_dir)

dim = DataImageMaker(data_dir, num_image, l = l)

i_list = []

print("Generate object image")
x_train, y_train = dim.gen_ori()
print("Save object image")
dim.save_image()

print("Generate object noizy image")
noize_x, noize_y = dim.gen_noize()
print("Save object noizy image")
dim.save_image(noize=True)

print("Run Hopfield")
hop = Hopfield_Restore(x_train, noize_x, noize_y, data_dir, model_name)
index, rest_ori, rest_noize, step_ori, step_noize = hop.calc()
i_list.append(index)
np.savetxt(os.path.join(save_dir, "rs_hop_epoch_{}_{}.csv".format(epoch, l)), np.array(i_list), delimiter=',')
np.save(os.path.join(save_dir, "rest_ori_epoch{}_seed_{}".format(epoch,l)), rest_ori)
np.save(os.path.join(save_dir, "rest_noize_epoch{}_seed_{}".format(epoch,l)), rest_noize)
np.save(os.path.join(save_dir, "step_ori_epoch{}_seed_{}".format(epoch,l)), step_ori)
np.save(os.path.join(save_dir, "step_noize_epoch{}_seed_{}".format(epoch,l)), step_noize)


