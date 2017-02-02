 from __future__ import division

import chainer
import chainer.functions as F
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
import chainer.links as L
import collections
import numpy as np

import six
    
class RFCNLENET(chainer.Chain):
    def __init__(self):
        """Initialize network architecture.
        Parameters
        ----------
        image_size: [int, int]
            The size of input image.
        """
        # feature extraction:
        # self.image_size = image_size
        super(RFCNLENET, self).__init__(
            conv1 = L.Convolution2D(1, 20, ksize=5, pad=5),
            conv2 = L.Convolution2D(20, 50, ksize=5),
            conv3 = L.Convolution2D(50, 500, ksize=3),
            conv4 = L.Convolution2D(500, 1, ksize=1),
            deconv = L.Deconvolution2D(in_channels=1, out_channels=2, ksize=10, stride=4, outsize=(28, 28)),
            gru = L.StatefulGRU(784, 784)
        )

        
    def reset_state(self):
        self.gru.reset_state()

    def __call__(self, x, ground_truth):
        """Forward RFCN Lenet pretrained model.
        This model has three losses:

        Parameters
        ----------
        x.data: (n_batch, 1, height, width)
            Input image.
        grand_truth: (n_batch, height, width)
            Grand truth of the object.
        """
        
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv1 = h

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv2 = h

        h = F.relu(self.conv3(h))
        self.h_conv3 = h

        h = self.conv4(h)
        self.h_conv4 = h

        h = self.deconv(h)
        self.deconv = h

        n_batch, _, _, _ = self.deconv.data.shape
        loss = None

        for i in six.moves.range(n_batch):
            y = self.gru(h[i,:,:,:])
            y = y[np.newaxis, :, :]
            gt = ground_truth[i,:][np.newaxis,:]
            a_loss = F.softmax_cross_entropy(y, gt)
            
            if loss is None:
                loss = a_loss
            else:
                loss += a_loss

        return loss
