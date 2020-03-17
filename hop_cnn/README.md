# Mobilenetv2 + Hopfield 

## environment

+ Linux: Ubuntu 18.04
+ GPU: GV100
+ using nvidia-docker: tensorflow:18.01-py3

## prepareation

Install the library.

``` sh
pip install -r requirments.txt
```

URL: https://github.com/DingKe/nn_playground/tree/master/xnornet
URL: https://github.com/unixpickle/hopfield

``` sh
git clone https://github.com/DingKe/nn_playground.git
git clone https://github.com/unixpickle/hopfield.git
```

Place "nn_playground / xnornet" and "hopfield / hopfield /" in each of the cloned folders under "hop_cnns/".

``` tree
hop_cnn-
        |xnornet/
        |hopfield/
                |network.py
                ...
        |hopfield_imagenet_1000/
        ...
        experiment_integration.py
```

## how to use

Specify the location of the image in the variable data_dir in experiment_integration.py.
The directory of the imag(ILSVRC2011: http://www.image-net.org/) of each class (1000 classes in this case) is placed in the specified directory.

``` tree
hogehoge-
        |class1-
                |class1_image.jpg
        |class2-
                |class2_image1.jpg
                |class2_image2.jpg
        ...
```

``` sh
python experiment_integration.py seed epoch_num
```

By running, you can perform a hopfield experiment using a model with the specified number of epochs.
At this time, set the model (saved_model / keras_mobilenetv2_freez_1000_epoch_ (epoch_num) .h5) that has been learned in advance for epoch_num.
Seed is required for image generation, and can have any value.
The result is generated in / result.

rs_hop_epoch_(epoch_num)\_(seed).csv :Discrimination performance when using mobilenet in the original image, performance after hopdield, noise image performance, performance after restoration

step_{noize,ori}_epoch(epoch_num)\_seed_(seed).npy :In the 3D numpy format, 1D indicates the number of update steps, 2D indicates the index representing the image, and 3D indicates the likelihood in each class.

rest_{noize,ori}_epoch(epoch_num)\_seed_(seed).npy :In the two-dimensional format, one dimension indicates the number of update steps, and two dimensions indicate the degree of restoration (cos similarity) of each image.

<!-- You can check each by result_viewer.ipynb -->

execute.sh is for performing multiple seeds experiments, and construct model. You can run the experiment at once by rewriting the contents.
If you are going to experiment from scratch, we recommend starting here．

``` sh
sh execute.sh
```
