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


def _init_weights():
    weights, biases = [], []

    unit_numbers = [COLUMN_NUMBER] + [UNIT_NUMBER] * HIDDEN_LAYER_NUMBER + [1]
    for input_unit_number, output_unit_number in zip(unit_numbers[:-1],
                                                     unit_numbers[1:]):
        # kaiming init 이다.
        limit = math.sqrt(6 / input_unit_number)

        weight = tf.Variable(
            tf.random_uniform([input_unit_number, output_unit_number], -limit,
                              limit))
        bias = tf.Variable(tf.zeros([output_unit_number]))
        weights.append(weight)
        biases.append(bias)

    return weights, biases


def _forward(x, weights, biases, training):
    assert len(weights) == len(biases)
    layer = x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias
        if i < len(biases) - 1:
            layer = tf.nn.leaky_relu(layer)
            layer = tf.layers.dropout(layer, rate=0.5, training=training)
    return layer


def init_models():
    x = tf.placeholder(tf.float32, [None, COLUMN_NUMBER], name='x')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    training = tf.placeholder(tf.bool)

    weights, biases = _init_weights()
    y_hat = _forward(x, weights, biases, training)
    loss = tf.reduce_mean(tf.square(y - y_hat))
    mae = tf.reduce_mean(tf.abs(y - y_hat))

    learning_rate = LEARNING_RATE

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, var_list=weights + biases)

    return {
        'x': x,
        'y': y,
        'training': training,
        'loss': loss,
        'mae': mae,
        'train_op': train_op
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

        _, batch_loss = session.run(
            (models['train_op'], models['loss']),
            feed_dict={
                models['x']: batch_x,
                models['y']: batch_y,
                models['training']: True,
            }, )

        if iter_idx % 200 == 0:
            train_loss = session.run(
                (models['mae']),
                feed_dict={
                    models['x']: train_x,
                    models['y']: train_y,
                    models['training']: False,
                }, )

            valid_loss = session.run(
                (models['mae']),
                feed_dict={
                    models['x']: valid_x,
                    models['y']: valid_y,
                    models['training']: False,
                }, )

            test_loss = session.run(
                (models['mae']),
                feed_dict={
                    models['x']: test_x,
                    models['y']: test_y,
                    models['training']: False,
                }, )

            if final_valid_loss > valid_loss:
                final_test_loss = test_loss
                final_valid_loss = valid_loss
                final_idx = iter_idx

            print('\n>> iter_idx:', iter_idx)
            print('>> train_loss:', train_loss)
            print('>> valid_loss:', valid_loss)
            print('>> test_loss:', test_loss)
            print('>> final_idx:', final_idx)
            print('>> final_valid_loss:', final_valid_loss)
            print('>> final_test_loss:', final_test_loss)

            if final_idx + EARLY_STOP_ITER < iter_idx:
                break


if __name__ == '__main__':
    main()
