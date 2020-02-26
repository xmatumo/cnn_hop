# args: epoch=60, mid_dim=10000, 2nd_dim=100 , lr=0.0003, batch_bool=1, init=1 defalt

# import library
import os
import sys
import pathlib
import numpy as np
import keras
import time
import glob
import random as rnd
from keras import optimizers, utils, losses, backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.applications.xception import Xception
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Add, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from binary_layers import BinaryDense, BinaryConv2D
from binary_ops import binary_tanh as binary_tanh_op
from keras.callbacks import LearningRateScheduler
from PIL import ImageFile
from PIL import Image

# fix bug
ImageFile.LOAD_TRUNCATED_IMAGES = True

def binary_tanh(x):
    return binary_tanh_op(x)

def mean_squared_error(y_true, y_pred):
    return K.abs(K.mean((y_pred), axis=-1)-1/2)

def generate_generator_multiple(generator, dir, dim_mid_layer,  batch_size, img_height,img_width, subset):
    genX1 = generator.flow_from_directory(dir,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          subset = subset,
                                          seed = 7)

    while True:
            X1i = genX1.next()
            null_input = np.zeros((X1i[0].shape[0], dim_mid_layer))
            yield [X1i[0], null_input], X1i[1]

dict_init = {"0": 'glorot_normal', "1":'he_normal'}

args = sys.argv

# parameta setting
batch_size = 32
epochs = int(args[1])
dim_mid_layer = int(args[2])
sec_dim = int(args[3])
lr = float(args[4])
bn = bool(int(args[5]))
init_dence = dict_init[args[6]]

lr_start = 1e-4
lr_end = 1e-5
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# input image dimensions
img_rows, img_cols = 250, 250
null_train = np.zeros((batch_size, dim_mid_layer))

# load image
data_dir = "/aist/data/imagenet_all/ILSVRC2011_images_train"
train_dir = pathlib.Path(data_dir)

# using image data generater 
train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)

train_generator = generate_generator_multiple(train_datagen, data_dir, dim_mid_layer, batch_size, 250, 250, 'training')

validation_generator = generate_generator_multiple(train_datagen, data_dir, dim_mid_layer, batch_size, 250, 250, 'validation')

# num of class
num_classes = 1000

# make models
result_dir = 'saved_models'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


# make model
def mobilenetv2_model_maker():

    input_tensor = Input(shape=(img_rows, img_cols, 3))
    mobile = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)

    x = Flatten()(mobile.output)
    x = BinaryDense(dim_mid_layer, H=1, use_bias=False)(x)
    x = Dropout(0.25)(x)
    if bn: x = BatchNormalization()(x)
    x = Activation(binary_tanh)(x)
    
    inputsub = Input(name='sub', shape=(dim_mid_layer,))
    x = Add(name='mid2')([x, inputsub])
   
    x = Dense(sec_dim, kernel_initializer=init_dence)(x)
    x = Dropout(0.25)(x)
    if bn: x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[input_tensor, inputsub], outputs = x)

    return model

if __name__ == '__main__':
    start = time.time()

    # make model
    model = mobilenetv2_model_maker()

    # freeze layers before last convolution layer
    for layer in model.layers[:-12]:
        layer.trainable = False

    # classify multiple classes
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adagrad(lr),
                metrics=['accuracy'])
    model.summary()
    
    print(int(np.ceil(len(list(train_dir.glob('**/*.JPEG')))*0.8)))
    
    # fit model
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch = int(np.ceil(len(list(train_dir.glob('**/*.JPEG')))*0.8)) // batch_size,
        validation_data = validation_generator, 
        validation_steps = int(np.ceil(len(list(train_dir.glob('**/*.JPEG')))*0.2)) // batch_size,
        epochs = epochs)
    
    
    result_dir = 'prog_loss'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
  
    # model name
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_mobilenetv2_freez_1000_epoch_{}.h5'.format(epochs)
    model_json_name = 'keras_mobilenetv2_freez_1000_epoch_{}.json'.format(epochs)


    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    json_model = model.to_json()
    open(os.path.join('saved_models',model_json_name), 'w').write(json_model)
#     model.save_weight(model_wight_name)

    print('Saved trained model at %s ' % model_path)
    