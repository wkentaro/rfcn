import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

import rfcn


class TestAssemble2D(unittest.TestCase):

    def setUp(self):
        self.ksize = 3
        self.x = numpy.tile(
            numpy.arange(self.ksize**2).reshape(1, self.ksize**2, 1, 1),
            (1, 1, self.ksize, self.ksize)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1,
            (1, 1, self.ksize, self.ksize)).astype(numpy.float32)
        self.check_backward_options = {'eps': 2.0 ** -8}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = rfcn.functions.assemble_2d(x, self.ksize)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        expect = numpy.arange(self.ksize**2).reshape(
            1, 1, self.ksize, self.ksize)
        testing.assert_allclose(expect, y_data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            rfcn.functions.Assemble2DFunction(self.ksize),
            x_data, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
