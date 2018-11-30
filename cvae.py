# -*- coding: utf-8 -*-

import os
import time
import gzip
import six
import numpy as np
import zhusuan as zs
import tensorflow as tf
from six.moves import urllib, range
from six.moves import cPickle as pickle
from skimage.exposure import rescale_intensity
from skimage import io, img_as_ubyte


# copy from examples.utils with dataset, save_image_collections, to_one_hot,download_dataset,load_mnist_realval

def save_image_collections(x, filename, shape=(10, 10), scale_each=False,
                           transpose=False):
    """
    :param shape: tuple
        The shape of final big images.
    :param x: numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param scale_each: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels),
        i.e., put channels behind.
    :return: `uint8` numpy array
        The output image.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.
    :param x: 1-D Numpy array of type int.
    :param depth: A int.
    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth))
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path)


def load_mnist_realval(path, one_hot=True, dequantify=False):
    """
    Loads the real valued MNIST dataset.

    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.
    :param dequantify:  Whether to add uniform noise to dequantify the data
        following (Uria, 2013).
    :return: The MNIST dataset.
    """

    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
                         '/mnist.pkl.gz', path)
    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    # t means labels
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    if dequantify:
        x_train += np.random.uniform(0, 1. / 256,
                                     size=x_train.shape).astype('float32')
        x_valid += np.random.uniform(0, 1. / 256,
                                     size=x_valid.shape).astype('float32')
        x_test += np.random.uniform(0, 1. / 256,
                                    size=x_test.shape).astype('float32')
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), \
           x_test, t_transform(t_test)


@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(x_dim, y, z_dim, n, n_particles=1):
    """
    build Bernoulli decoder network
    :param x_dim: x dimension
    :param y: the class label
    :param z_dim: z dimension
    :param n: batch size
    :param n_particles:
    :return: BayesianNet
    """
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim], dtype=tf.float32)
    z = bn.normal("z", z_mean, std=1., group_ndims=1, n_samples=n_particles)
    z = tf.reshape(z, (n, z_dim))
    z = tf.concat([z, y], 1)
    h = tf.layers.dense(z, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)
    bn.output("x_mean", tf.sigmoid(x_logits))
    bn.bernoulli("x", x_logits, group_ndims=1)
    return bn


@zs.reuse_variables(scope="q_net")
def build_q_net(x, y, z_dim, n_z_per_x):
    """
    build Gaussian Encoder
    :param x: input matrix with values 0 or 1
    :param y: the class label
    :param z_dim: z dimension
    :param n_z_per_x:
    :return: BayesianNet
    """
    bn = zs.BayesianNet()
    x = tf.concat([tf.to_float(x), y], 1)
    h = tf.layers.dense(tf.to_float(x), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_z_per_x)
    return bn


def main():
    # Load MNIST
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        load_mnist_realval("./mnist.pkl.gz")
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([t_train, t_valid])
    x_dim = x_train.shape[1]
    y_dim = y_train.shape[1]
    train_data = np.hstack([x_train, y_train])
    data_size = train_data.shape[0]

    # Define model parameters
    z_dim = 40

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    x = tf.to_int32(tf.less(tf.random_uniform(tf.shape(x_input)), x_input))
    y_input = tf.placeholder(tf.float32, shape=[None, y_dim], name="y")
    n = tf.placeholder(tf.int32, shape=[], name="n")

    meta_model = build_gen(x_dim, y_input, z_dim, n, n_particles)
    variational = build_q_net(x, y_input, z_dim, n_particles)

    lower_bound = zs.variational.elbo(
        meta_model, {"x": x}, variational=variational, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)

    # Random generation
    x_gen = tf.reshape(meta_model.observe()["x_mean"], [-1, 28, 28, 1])

    # Define training/evaluation parameters
    epochs = 3000
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    save_freq = 10
    result_path = "results/cvae"
    condition = np.array(range(10))
    condition_onehot = to_one_hot(condition, 10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(train_data)
            x_train = train_data[:, :x_dim].reshape((data_size, x_dim))
            y_train = train_data[:, x_dim:].reshape((data_size, y_dim))
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch, y_input: y_batch,
                                            n_particles: 1, n: batch_size})
                lbs.append(lb)
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % save_freq == 0:
                for item in range(10):
                    test_input = np.tile(condition_onehot[item], [100, 1])
                    images = sess.run(x_gen, feed_dict={y_input: test_input, n: 100, n_particles: 1})
                    name = os.path.join(result_path + '/num_{}/'.format(item),
                                        "cvae_epoch_{}.png".format(epoch))
                    save_image_collections(images, name)


if __name__ == "__main__":
    main()
