import os

import numpy as np
import xgboost as xgb

EXP_NUMBER = 6
# TRAIN = False
TRAIN = True
MULTI_TASK = False
FINE_TUNE = False
# fine tune을 하려면 multi-task 여야 합니다.
assert not FINE_TUNE or (MULTI_TASK and FINE_TUNE)

if MULTI_TASK:
    if FINE_TUNE:
        MODEL_PATH = "tmp/xgbmodel-{}-fine".format(EXP_NUMBER)
    else:
        MODEL_PATH = "tmp/xgbmodel-{}-multi".format(EXP_NUMBER)
else:
    MODEL_PATH = "tmp/xgbmodel-{}-2".format(EXP_NUMBER)


def _split_x_y(data):
    column_number = data.shape[1] - 1
    return data[:, :column_number], data[:, column_number:]


def _get_batch(train_x, train_y, batch_size):
    batch_i = np.random.choice(train_y.shape[0], batch_size)
    batch_x = train_x[batch_i, :]
    batch_y = train_y[batch_i, :]
    return batch_x, batch_y


def _make_multitask(raw):
    raw = np.concatenate(
        (np.ones((raw.shape[0], 1)) * EXP_NUMBER, raw), axis=1)
    return raw


def _get_data(filename):
    raw = np.genfromtxt(filename, delimiter=',')
    return raw


def _get_error(model, x, y):
    i = xgb.DMatrix(x, missing=np.nan, nthread=1)
    o = model.predict(i)
    y = np.reshape(y, (-1, ))
    error = np.mean(np.absolute(o - y))
    return error


def main():
    if MULTI_TASK and not FINE_TUNE:
        data_train = _get_data('data/{}/train.csv'.format(0))
    else:
        data_train = _get_data('data/{}/train.csv'.format(EXP_NUMBER))
    train_x, train_y = _split_x_y(data_train)
    if FINE_TUNE:
        train_x = _make_multitask(train_x)

    data_valid = _get_data('data/{}/valid.csv'.format(EXP_NUMBER))
    valid_x, valid_y = _split_x_y(data_valid)
    if MULTI_TASK:
        valid_x = _make_multitask(valid_x)

    data_test = _get_data('data/{}/test.csv'.format(EXP_NUMBER))
    test_x, test_y = _split_x_y(data_test)
    if MULTI_TASK:
        test_x = _make_multitask(test_x)

    train_mat = xgb.DMatrix(train_x, label=train_y, missing=np.nan, nthread=30)
    valid_mat = xgb.DMatrix(valid_x, label=valid_y, missing=np.nan, nthread=30)

    params = dict(
        n_jobs=os.cpu_count() - 1,
        silent=0,
        missing=np.nan,
        eval_metric="mae",
        tree_method="gpu_hist",
        booster="gbtree",
        max_bin=25600,
        max_depth=15,
        colsample_bytree=0.7,
        subsample=0.7,
        learning_rate=0.01,
        gamma=10, )
    if FINE_TUNE:
        params['model_in'] = 'tmp/xgbmodel-0'


    if TRAIN:
        # if FINE_TUNE:
        #     model = xgb.train(
        #         params,
        #         train_mat,
        #         num_boost_round=10000,
        #         evals=[(train_mat, "train"), (valid_mat, "valid")],
        #         early_stopping_rounds=50,
        #         verbose_eval=True)
        # else:
        model = xgb.train(
            params,
            train_mat,
            num_boost_round=10000,
            evals=[(train_mat, "train"), (valid_mat, "valid")],
            early_stopping_rounds=50,
            verbose_eval=True)
        model.save_model(MODEL_PATH)

    model = xgb.Booster(params)  # init model
    model.load_model(MODEL_PATH)

    train_error = _get_error(model, train_x, train_y)
    print('>> train_error', train_error)
    valid_error = _get_error(model, valid_x, valid_y)
    print('>> valid_error', valid_error)
    test_error = _get_error(model, test_x, test_y)
    print('>> test_error', test_error)

    # # import matplotlib.pyplot as plt
    # # plt.scatter(train_x[:, -1], train_y)
    # # plt.savefig('books_read.png')
    # # assert False


if __name__ == '__main__':
    main()
