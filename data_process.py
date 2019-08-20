import random
import math

UNIT_NUMBER = 10
EXP_NUMBER = 6
BASE_PATH = 'data/{}/'.format(EXP_NUMBER)


def main():
    rows = []
    f = open(BASE_PATH + 'raw.txt', 'r')
    for line in f:
        cols = line.strip().split('  ')
        # if cols[0] != '0':
        #     continue
        row = [float(cols[0]), float(cols[1])
               ] + [0] * 2 * UNIT_NUMBER + [float(cols[2])]
        rows.append(row)
    f.close()

    # 속도 계산
    for i, row in enumerate(rows):
        for j in range(2, 2 + UNIT_NUMBER):
            delta_i = j
            if i < delta_i:
                continue
            try:
                rows[i][j] = (rows[i][1] - rows[i - delta_i][1]) / (
                    rows[i][0] - rows[i - delta_i][0])
            except:
                rows[i][j] = float('nan')

    # 가속도 계산
    for i, row in enumerate(rows):
        for j in range(2 + UNIT_NUMBER, 2 + 2 * UNIT_NUMBER):
            delta_i = j - UNIT_NUMBER
            if i < delta_i + UNIT_NUMBER:
                continue
            try:
                rows[i][j] = (rows[i][j - UNIT_NUMBER] -
                              rows[i - delta_i][j - UNIT_NUMBER]) / (
                                  rows[i][0] - rows[i - delta_i][0])
            except:
                rows[i][j] = float('nan')

    f_train = open(BASE_PATH + 'train.csv', 'w')
    f_valid = open(BASE_PATH + 'valid.csv', 'w')
    f_test = open(BASE_PATH + 'test.csv', 'w')
    for row in rows:
        line = ','.join(str(t) if not math.isnan(t) else ''
                        for t in row[1:]) + '\n'
        if random.random() < 0.9:
            if random.random() < 0.02:
                f_valid.write(line)
            else:
                f_train.write(line)
        else:
            f_test.write(line)
    f_valid.close()
    f_train.close()
    f_test.close()


if __name__ == '__main__':
    main()
