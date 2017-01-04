# !/usr/bin/python 

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from rfcn.external.faster_rcnn.models.rpn import RPN
from rfcn import utils
from rfcn.models.fcis_vgg16 import FCISVGG16


class CONV_FCIS(chainer.Chain):
    def __init__(self, C, k=7):
        super(CONV_FCIS, self).__init__(
            conv_1 = L.Convolution2D(2*(k**2), k, 11, pad=5),
            conv_2 = L.Convolution2D(k, 1, 11, pad=5),

            fcis = FCISVGG16(C, k)
        )

        self.ksize = k
        self.C = C

    def __call__(self, x, t_label_cls, t_label_inst):
        # TODO(mukuact): need to load pre-trained weight
        self.fcis(x) 

        # (b_batch, 2 * k^2 * (C + 1), height/32, width/32)
        score = self.fcis.score
        
        sc_shape = score.data.shape
        
        # get k**2 score map for 'ic'th category
        # and apply conv func to them
        # filter = roi
        iks = np.arange(0, 2*(C+1)*(k**2), (C+1))   
        cls_likelihood = []
        for ic in xrange(C+1):
            a_cls_score = F.get_item(score, np.s_[:, iks+ic])
            h = F.relu(conv_1(a_cls_score))
            a_cls_likelihood = conv_2(h)
            cls_likelihood.append(a_cls_likelihood)

        # (n_batch, C+1, height/32, width/32)
        cls_likelihood = F.concat(cls_likelihood, axis=1)

            #
            hoge(a_cls_likelihood)
            # roi's grand truth is assumed as center of roi
            # grand truth is seemed to have a same resolution as the score map 
            a_loss_seg = F.softmax_cross_entropy(cat_likelihood, t_label_cls)
             


