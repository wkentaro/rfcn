from __future__ import division

import collections

import chainer
import chainer.functions as F
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
import chainer.links as L
import numpy as np

from rfcn import utils
import six


class VGG16Trunk(chainer.Chain):

    def __init__(self):
        super(VGG16Trunk, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
        )

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv1 = h  # 1/2

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv2 = h  # 1/4

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv3 = h  # 1/8

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv4 = h  # 1/16

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv5 = h  # 1/32

        return self.h_conv5

class ConvGRU(chainer.Chain):
    """Convolution Gated Recurrent Unit function (ConvGRU).

    Convolution GRU function has six parameters :math:`W_r`, :math:`W_z`,
    :math:`W`, :math:`U_r`, :math:`U_z`, and :math:`U`.
    All these parameters are :math:`n \\times n` matrices,
    where :math:`n` is the dimension of hidden vectors.

    Given input vector :math:`x`, Stateful GRU returns the next
    hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r \ast x + U_r \ast h), \\\\
       z &=& \\sigma(W_z \ast x + U_z \ast h), \\\\
       \\bar{h} &=& \\tanh(W \ast x + U \ast (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`h` is current hidden vector.

    Args:
        in_size(int): Dimension of input vector :math:`x`.
        out_size(int): Dimension of hidden vector :math:`h`.
        init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the
            GRU's input units (:math:`W`). Maybe be `None` to use default
            initialization.
        inner_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the GRU's inner
            recurrent units (:math:`U`).
            Maybe be ``None`` to use default initialization.
        bias_init: A callable or scalar used to initialize the bias values for
            both the GRU's inner and input units. Maybe be ``None`` to use
            default initialization.
    """

    def __init__(self, n_channels, ksize):
        super(ConvGRU, self).__init__(
            W_r=L.Convolution2D(n_channels, n_channels, ksize, stride=1, pad=1),
            U_r=L.Convolution2D(n_channels, n_channels, ksize, stride=1, pad=1),
            W_z=L.Convolution2D(n_channels, n_channels, ksize, stride=1, pad=1),
            U_z=L.Convolution2D(n_channels, n_channels, ksize, stride=1, pad=1),
            W=L.Convolution2D(n_channels, n_channels, ksize, stride=1, pad=1),
            U=L.Convolution2D(n_channels, n_channels, ksize, stride=1, pad=1)
        )

    def to_cpu(self):
        super(ConvGRU, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(ConvGRU, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == numpy:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        z = self.W_z(x)
        h_bar = self.W(x)
        if self.h is not None:
            r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.h))
            z += self.U_z(self.h)
            h_bar += self.U(r * self.h)
        z = sigmoid.sigmoid(z)
        h_bar = tanh.tanh(h_bar)

        h_new = z * h_bar
        if self.h is not None:
            h_new += (1 - z) * self.h
        self.h = h_new
        return self.h
    
class RFCNVGG16(chainer.Chain):
    
    """RFCN based on pretrained model of VGG16."""
    
    def __init__(self, image_size):
        """Initialize network architecture.
        Parameters
        ----------
        image_size: [int, int]
            The size of input image.
        """
        super(RFCNVGG16, self).__init__()
        self.image_size = image_size

        # feature extraction:
        self.add_link('trunk', VGG16Trunk())

        self.add_link('conv6',
                      L.Convolution2D(512, 128, ksize=1))
        
        # Convolutional GRU:
        self.add_link('conv_gru', ConvGRU(n_channels=128, ksize=3))

        # Convert to gray image:
        self.add_link('conv7', L.Convolution2D(128, 1, ksize=1))

        # Upsampling:
        self.add_link('deconv',
                      L.Deconvolution2D(in_channels=1, out_channels=1, ksize=20, stride=8, outsize=image_size))

    def reset_state(self):
        self.conv_gru.reset_state()

    def __call__(self, x, ground_truth):
        """Forward RFCN with VGG16 pretrained model.
        This model has three losses:
        - label_fg_loss:
            a softmax segmentation loss 1 over the foreground mask
            of the ground-truth category (if roi is positive).
        - rpn_loss_bbox:
            a bbox regression loss (if roi is positive).
        Parameters
        ----------
        x.data: (n_batch, 3, height, width)
            Input image.
        grand_truth: (n_batch, height, width)
            Grand truth of the object.
        """
        
        assert self.image_size == ground_truth.shape[1:]
        # feature extraction
        # ---------------------------------------------------------------------
        # (n_batch, 3, height/32, width/32)
        h_conv5 = self.trunk(x)
        # ---------------------------------------------------------------------

        h_conv6 = self.conv6(h_conv5)
        
        # im_info: [[height, width, image_scale], ...]
        n_batch, _, height_32, width_32 = h_conv5.data.shape
        im_info = np.array([[height_32, width_32, 1]], dtype=np.float32)
        im_info = np.repeat(im_info, n_batch, axis=0)

        for i in six.moves.range(n_batch):
            h_conv_gru = self.conv_gru(h_conv6[i,:,:,:][np.newaxis,:])
            h_conv7 = self.conv7(h_conv_gru)
            h_deconv = self.deconv(h_conv7)
            a_loss = F.softmax_cross_entropy(h_deconv, ground_truth[i])
            if loss is None:
                loss = a_loss
            else:
                loss += a_loss

        return loss
