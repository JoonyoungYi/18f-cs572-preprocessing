import random
import math

UNIT_NUMBER = 10
BASE_PATH = 'data/simulate.csv'


def main():
    rows = []
    for t in range(1, 3150):
        row = [t / 1000, 60 * 3150 / t ] + [0] * 2 * UNIT_NUMBER
        rows.append(row)
        print(t)
    
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

    f = open(BASE_PATH, 'w')
    for row in rows:
        line = ','.join(str(t) if not math.isnan(t) else ''
                        for t in row[1:]) + '\n'
        f.write(line)
    f.close()


if __name__ == '__main__':
    main()
