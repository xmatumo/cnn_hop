import numpy as np
import tensorflow as tf
from hopfield import Network, covariance_update, extended_storkey_update,hebbian_update


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

### Hopfield model 

#### If you use other update method, uncomment the following

# def train(sess, network, dataset, mid_layer_dim):
#     """
#     Train the Hopfield network.
#     """
#     images_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, mid_layer_dim))
#     update = covariance_update(tf.greater_equal(images_ph, 0.0), network.weights, network.thresholds)
# #     update = hebbian_update(tf.greater_equal(images_ph, 0.0), network.weights)

#     sess.run(tf.global_variables_initializer())
#     for i in range(0, len(dataset), BATCH_SIZE):
#         images = dataset[i : i+BATCH_SIZE]
#         sess.run(update, feed_dict={images_ph: images})


def train(sess, network, dataset, mid_layer_dim):
   # train hopfield for encoded image by CNN
    image_ph = tf.placeholder(tf.float32, shape=(mid_layer_dim,))
    joined = tf.greater_equal(image_ph, 0.0)
    update = extended_storkey_update(joined, network.weights)
    sess.run(tf.global_variables_initializer())
    i = 0
    for image in dataset:
        sess.run(update, feed_dict={image_ph: image})


def iterate_network(sess, network, images, num):
    # output setp of hopfield training
    iter_step = []
    images_ph = tf.placeholder(tf.float32, shape=images.shape)
    output = network.step(images_ph)
    for i in range(num):
        images = sess.run(output, feed_dict={images_ph: images})
        iter_step.append(images)
    return iter_step