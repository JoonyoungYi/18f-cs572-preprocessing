def add_to_f_write(f_write, idx, kind):
    f_read = open('data/{}/{}.csv'.format(idx, kind), 'r')
    for line in f_read:
        f_write.write('{},'.format(idx) + line)


for kind in ['test', 'train', 'valid']:
    print('>> kind:', kind)

    f_write = open('data/0/{}.csv'.format(kind), 'w')
    for idx in range(1, 7):
        add_to_f_write(f_write, idx, kind)
    f_write.close()
