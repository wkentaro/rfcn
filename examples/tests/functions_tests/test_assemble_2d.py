import chainer
import numpy as np

import rfcn


if __name__ == '__main__':
    x_data = np.zeros((1, 9, 9, 9), dtype=np.float32)
    for i in xrange(9):
        x_data[:, i, :, :] = i
    x = chainer.Variable(x_data)
    y = rfcn.functions.assemble_2d(x, ksize=3)
