import math

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

EXP_NUMBER = 6
COLUMN_NUMBER = 21
UNIT_NUMBER = 200
HIDDEN_LAYER_NUMBER = 3
LEARNING_RATE = 1e-3
# NORMALIZING = 10
NORMALIZING = 0
EARLY_STOP_ITER = 2000


def init_models():
    x = tf.placeholder(tf.float32, [None, COLUMN_NUMBER], name='x')
    y = tf.placeholder(tf.float32, [None, 1], name='y')

    a = tf.Variable(tf.random_uniform([COLUMN_NUMBER, 1], -1, 1))
    b = tf.Variable(tf.zeros([1, 1]))

    y_hat = tf.reshape(tf.matmul(x, a) + b, (-1, 1))
    loss = tf.reduce_mean(tf.square(y - y_hat))
    mae = tf.reduce_mean(tf.abs(y - y_hat))

    learning_rate = LEARNING_RATE

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, var_list=[a, b])

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'mae': mae,
        'train_op': train_op,
        'a': a,
        'b': b
    }


def _split_x_y(data):
    return data[:, :COLUMN_NUMBER], data[:, COLUMN_NUMBER:]


def _get_batch(train_x, train_y, batch_size):
    batch_i = np.random.choice(train_y.shape[0], batch_size)
    batch_x = train_x[batch_i, :]
    batch_y = train_y[batch_i, :]
    return batch_x, batch_y


def _get_data(filename):
    raw = genfromtxt('data/{}/'.format(EXP_NUMBER) + filename, delimiter=',')
    raw[:, 1] = raw[:, 1]
    raw[:, -1] = raw[:, -1]
    raw[np.isnan(raw)] = 0
    return raw


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models()
    session.run(tf.global_variables_initializer())

    data_train = _get_data('train.csv')
    train_x, train_y = _split_x_y(data_train)
    data_valid = _get_data('valid.csv')
    valid_x, valid_y = _split_x_y(data_valid)
    data_test = _get_data('test.csv')
    test_x, test_y = _split_x_y(data_test)

    # import matplotlib.pyplot as plt
    # plt.scatter(train_x[:, -1], train_y)
    # plt.savefig('books_read.png')
    # assert False

    k = 0.0
    final_idx = 0
    final_valid_loss = float('inf')
    final_test_loss = float('inf')
    for iter_idx in range(1, 1000000):
        batch_x, batch_y = _get_batch(train_x, train_y, 1000)

        _, batch_loss, a, b = session.run(
            (models['train_op'], models['loss'], models['a'], models['b']),
            feed_dict={
                models['x']: batch_x,
                models['y']: batch_y,
            }, )

        if iter_idx % 200 == 0:
            train_loss = session.run(
                (models['mae']),
                feed_dict={
                    models['x']: train_x,
                    models['y']: train_y,
                }, )

            valid_loss = session.run(
                (models['mae']),
                feed_dict={
                    models['x']: valid_x,
                    models['y']: valid_y,
                }, )

            test_loss = session.run(
                (models['mae']),
                feed_dict={
                    models['x']: test_x,
                    models['y']: test_y,
                }, )

            if final_valid_loss > valid_loss:
                final_test_loss = test_loss
                final_valid_loss = valid_loss
                final_idx = iter_idx
                np.save('tmp/bw-{}-a.npy'.format(EXP_NUMBER), a)
                np.save('tmp/bw-{}-b.npy'.format(EXP_NUMBER), b)

            print('\n>> iter_idx:', iter_idx)
            print('>> train_loss:', train_loss)
            print('>> valid_loss:', valid_loss)
            print('>> test_loss:', test_loss)
            print('>> final_idx:', final_idx)
            print('>> final_valid_loss:', final_valid_loss)
            print('>> final_test_loss:', final_test_loss)
            print(a)
            print(b)

            if final_idx + EARLY_STOP_ITER < iter_idx:
                break


if __name__ == '__main__':
    main()
