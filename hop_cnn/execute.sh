python hopfield_imagenet_1000/mobilenet_binary_allcls.py 60 10000 100 0.0003 1 1;


do
    for i in 21 31 41 51 61 71 81 91 101 111
    do
        python experiment_integration.py $i 60
    done
done