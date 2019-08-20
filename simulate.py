import os
import random
import math

import numpy as np
import xgboost as xgb

UNIT_NUMBER = 10
rows = []
for t in range(0, 21):
    rows.append([t / 3150, 0] + [0] * 2 * UNIT_NUMBER + [0])

a = np.load('tmp/bw-1-a.npy')
b = np.load('tmp/bw-1-b.npy')


def _bw(x):
    # y = np.dot(x.flatten(), a.flatten()) + b
    # # print(x)
    # # print(a)
    # # print(b)
    # y = max(0, min(y, 60))
    y = max(0, x[0] - 6)  + random.random() * 1
    return y


MODEL_PATH = "tmp/xgbmodel-{}".format(1)
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
model = xgb.Booster(params)  # init model
model.load_model(MODEL_PATH)


def _xgb(x):
    i = xgb.DMatrix(x, missing=np.nan, nthread=1)
    o = model.predict(i)
    y = np.reshape(o, (-1, ))
    return y[0]


def _get_row(t, theta_d_new):
    i = t

    row = [t / 3500, theta_d_new] + [0] * 2 * UNIT_NUMBER + [0]
    # theta_d = 60 * t / 3150

    # 속도 계산
    for j in range(2, 2 + UNIT_NUMBER):
        delta_i = j
        if i < delta_i:
            continue

        # print(row[1])
        # print(rows[i - delta_i][1])
        # print(row[0])
        # print(rows[i - delta_i][0])
        row[j] = (row[1] - rows[i - delta_i][1]) / (
            row[0] - rows[i - delta_i][0])

    # 가속도 계산
    for j in range(2 + UNIT_NUMBER, 2 + 2 * UNIT_NUMBER):
        delta_i = j - UNIT_NUMBER
        if i < delta_i + UNIT_NUMBER:
            continue

        row[j] = (row[j - UNIT_NUMBER] - rows[i - delta_i][j - UNIT_NUMBER]
                  ) / (row[0] - rows[i - delta_i][0])
    return row


def _simulate_for_timestamp_t(t, theta_d_new):
    global rows

    assert t >= 21 and t <= 3521

    row = _get_row(t, theta_d_new)

    assert len(rows) == t
    rows.append(row)

    x = np.array([rows[t][1:-1]])
    y = _xgb(x)

    # x = np.array(rows[t][1:-1])
    # y = _bw(x)

    rows[t][-1] = y
    print(y)


def _get_theta_i(theta_d, theta_i):
    for _ in range(4):
        print(theta_i)
        row = _get_row(t, theta_i)
        # print(row)

        # x = np.array([row[1:-1]])
        # theta_y = _xgb(x)

        x = np.array(row[1:-1])
        theta_y = _bw(x)
        print(theta_y)

        print(theta_d)


        if theta_d < theta_y:
            theta_i -= 0.01
        else:
            theta_i += 0.01
            # print('plus!')
        # input('')

    return theta_i


theta_ds = [0] * 21
theta_os = [0] * 21

for t in range(21, 1000):
    # for t in range(21, 100):
    print('>>', t)
    theta_d = (t - 20) * 0.02
    print('.. theta_d:', theta_d)

    theta_i = _get_theta_i(theta_d, rows[t - 1][1])
    print('.. theta_i:', theta_i)
    # theta_i = theta_d
    _simulate_for_timestamp_t(t, theta_i)

    # print(rows)
    theta_ds.append(theta_d)

    theta_o = max(0, theta_d - 6) + random.random() * 0.3
    theta_os.append(theta_o)

    if t % 50 == 0:
        import matplotlib.pyplot as plt
        x = np.array(rows)
        plt.scatter(theta_ds, theta_ds)
        plt.scatter(theta_ds, theta_os)
        plt.scatter(theta_ds, x[0:, -1])

        plt.savefig('cache/sim.png')
        plt.clf()

assert False

# EXP_NUMBER = 0

#
#
# def _split_x_y(data):
#     column_number = data.shape[1] - 1
#     return data[:, :column_number], data[:, column_number:]
#
#
# def _get_data():
#     raw = np.genfromtxt('data/simulate.csv', delimiter=',')
#     return raw
#
#
# def main():
#     data = _get_data()
#     train_x, train_y = _split_x_y(data)
#
#     train_mat = xgb.DMatrix(train_x, label=train_y, missing=np.nan, nthread=30)
#

if __name__ == '__main__':
    main()
