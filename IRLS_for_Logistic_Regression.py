# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_svmlight_file


def preprocess(X_train, X_test, y_train, y_test):
    # add a dimension of ones to X to simplify computation,then the dimension is 124
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    X_train = np.hstack((np.ones((N_train, 1)), X_train.toarray()))
    X_test = np.hstack((np.ones((N_test, 1)), X_test.toarray()))
    y_train = y_train.reshape((N_train, 1))
    y_test = y_test.reshape((N_test, 1))

    # Transfet the label to 0, 1
    y_train = np.where(y_train == -1, 0, 1)
    y_test = np.where(y_test == -1, 0, 1)
    return X_train, X_test, y_train, y_test

def log_likelihood(w, X, y, L2_lamda=0):
    # Also is the Cross-Entropy function
    res = tf.matmul(tf.matmul(tf.transpose(w), tf.transpose(X)), y) - tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(X, w))))
    if L2_lamda > 0:
        res += -0.5 * L2_lamda * tf.matmul(tf.transpose(w), w)
    return -res[0][0]

def predict_proba(X, w):
    # The softmax function
    y = tf.constant(np.array([0., 1.]), dtype=tf.float32)
    proba = tf.exp(tf.matmul(X, w) * y) / (1 + tf.exp(tf.matmul(X, w)))
    return proba

def score(X, y, w):
    p = predict_proba(X, w)
    y_pred = tf.cast(tf.argmax(p, axis=1), tf.float32)
    y = tf.squeeze(y)
    acc = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
    return acc

def optimize(w, w_update):
    return w.assign(w - w_update)

def update(w, X, y, L2_lamda=0):
    mul = tf.sigmoid(tf.matmul(X, w))
    R_flat = mul * (1 - mul)

    dim = X.shape.as_list()[1]
    L2_reg_term = L2_lamda * tf.eye(dim)
    XRX = tf.matmul(tf.transpose(X), R_flat * X) + L2_reg_term
    S, U, V = tf.svd(XRX, full_matrices=True, compute_uv=True)
    S = tf.expand_dims(S, 1)
    S_pinv = tf.where(tf.not_equal(S, 0), 1 / S, tf.zeros_like(S))
    XRX_pinv = tf.matmul(V, S_pinv * tf.transpose(U))

    w_update = tf.matmul(XRX_pinv, tf.matmul(tf.transpose(X), mul - y) + L2_lamda * w)
    return w_update


if __name__ == '__main__':
    X_train, y_train = load_svmlight_file('a9a', n_features=123, dtype=np.float32)
    X_test, y_test = load_svmlight_file('a9a.t', n_features=123, dtype=np.float32)
    X_train, X_test, y_train, y_test =preprocess(X_train, X_test, y_train, y_test)

    L2_lamda = 10  # The L2 norm parameter lamda

    N, dim = X_train.shape
    X = tf.placeholder(dtype=tf.float32, shape=(None, 124), name="X")
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

    w = tf.Variable(0.01 * tf.ones((dim, 1), dtype=tf.float32), name='w')
    w_update = update(w, X, y, L2_lamda)
    loss = log_likelihood(w, X, y, L2_lamda)
    acc = score(X, y, w)
    optimize_op = optimize(w, w_update)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())

    max_iter = 100
    for i in range(1, max_iter):
        print('iter: {}'.format(i))

        print('   log likelihood: {}'.format(session.run(loss, feed_dict={X: X_train, y: y_train})))
        train_acc = session.run(acc, feed_dict={X: X_train, y: y_train})
        test_acc = session.run(acc, feed_dict={X: X_test, y: y_test})
        print('   train accuracy: {}, test accuracy: {}'.format(train_acc, test_acc))
        L2_norm_w = np.linalg.norm(session.run(w))
        print('   L2 norm of w: {}'.format(L2_norm_w))
        deri_w = np.linalg.norm(session.run(w_update, feed_dict={X: X_train, y: y_train}))
        print('   derivative of w_old and w: {}'.format(deri_w))

        if deri_w < 0.001:
            break
        w_new = session.run(optimize_op, feed_dict={X: X_train, y: y_train})

    print('Done.')
