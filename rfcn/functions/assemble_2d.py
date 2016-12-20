from chainer import cuda
from chainer.function import Function
from chainer.utils import type_check
import numpy as np


class Assemble2DFunction(Function):

    def __init__(self, ksize):
        if isinstance(ksize, int):
            self.ksize = ksize
        else:
            raise TypeError('Integer type is only supported for ksize.\n'
                            'Actual: {}'.format(type(ksize)))

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)

        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x = inputs[0]
        n, c, h, w = x.shape
        kh = h // self.ksize
        kw = w // self.ksize
        kc = c // (self.ksize**2)
        y = xp.zeros((n, kc, h, w), dtype=np.float32)
        for ik in xrange(self.ksize**2):
            y1 = kh * ik
            y2 = y1 + kh
            x1 = kw * ik
            x2 = x1 + kw
            c1 = kc * ik
            c2 = c1 + kc
            y[:, :, y1:y2, x1:x2] = x[:, c1:c2, y1:y2, x1:x2]
        return y,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x = inputs[0]
        gy = grad_outputs[0]
        out_n, out_h, out_w = gy.shape
        n, c, h, w = x.shape
        assert out_n == n
        assert out_h == h
        assert out_w == w
        kh = h // self.ksize
        kw = w // self.ksize
        kc = c // (self.ksize**2)
        gx = xp.zeros_like(x)
        for ik in xrange(self.ksize**2):
            y1 = kh * ik
            y2 = y1 + kh
            x1 = kw * ik
            x2 = x1 + kw
            c1 = kc * ik
            c2 = c1 + kc
            gx[:, c1:c2, y1:y2, x1:x2] = gy[:, y1:y2, x1:x2]
        return gx,


def assemble_2d(x, ksize):
    return Assemble2DFunction(ksize)(x)
