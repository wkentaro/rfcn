# !/usr/bin/python
import PIL

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from rfcn import functions
from rfcn.models.fcis_vgg16 import VGG16Trunk


class CONV_FCIS(chainer.Chain):
    def __init__(self, C, k=7):
        super(CONV_FCIS, self).__init__(
            trunk=VGG16Trunk(),
            conv_score=L.Convolution2D(512, 2 * k**2 * (C + 1), ksize=1),

            conv_1=L.Convolution2D(2*(k**2), k, 11, pad=5),
            conv_2=L.Convolution2D(k, 1, 11, pad=5),

        )

        self.k = k
        self.C = C

    def __call__(self, x, t_label_cls, t_label_inst):
        """
        Parameters
        ----------
        x.data: (n_batch, 3, height, width)
            Input image.
        t_label_cls: (n_batch, height, width)
            Label image about object class.
        t_label_inst: (n_batch, height, width)
            Label image about object instances.
        """
        xp = chainer.cuda.get_array_module(x)
        C = self.C
        k = self.ksize

        # TODO(mukuact): need to load pre-trained weight
        h_conv5 = self.trunk(x)
        # (b_batch, 2 * k^2 * (C + 1), height/32, width/32)
        score = self.conv_score(h_conv5)

        sc_shape = score.data.shape

        _, _, height_32s, width_32s = sc_shape
        # grand truth
        t_label_inst_data = chainer.cuda.to_cpu(t_label_inst.data)
        t_label_inst_pil = PIL.Image.fromarray(t_label_inst_data[0])
        t_label_inst_32s = t_label_inst_pil.resize((width_32s, height_32s))
        t_label_inst_32s = np.array(t_label_inst_32s)

        t_label_cls_data = chainer.cuda.to_cpu(t_label_cls.data)
        t_label_cls_pil = PIL.Image.fromarray(t_label_cls_data[0])
        t_label_cls_32s = t_label_cls_pil.resize((width_32s, height_32s))
        t_label_cls_32s = np.array(t_label_cls_32s)

        # get k**2 score map for 'ic'th category
        # and apply conv func to them
        # iks:[0, C+1+1, 2(C+1)+1, 3(C+1)+1, ...]
        iks = np.arange(0, 2*(C+1)*(k**2), (C+1))
        cls_likelihood = []
        for ic in xrange(C+1):
            # applying same convolution filter to the maps of each category
            a_cls_score = F.get_item(score, np.s_[:, iks+ic])
            h = F.relu(self.conv_1(a_cls_score))
            a_cls_likelihood = self.conv_2(h)
            cls_likelihood.append(a_cls_likelihood)

        # (n_batch, C+1, height/32, width/32)
        cls_likelihood = F.concat(cls_likelihood, axis=1)

        # roi's grand truth is assumed as center of roi
        # grand truth is seemed to have a same resolution as the score map
        loss_cls = F.softmax_cross_entropy(cls_likelihood, t_label_cls_32s)

        # get rois according to cls score
        # class is true and score > 0.7
        rois = self._get_rois(cls_likelihood, t_label_cls_32s)
        loss_seg = chainer.Variable(xp.array(0, dtype=np.float32),
                                    volatile='auto')
        for roi in rois:
            cls_id, x1, y1, x2, y2 = roi
            roi_h = y2 - y1
            roi_w = x2 - x2

            # score map in ROI: (n_batch=1, 2 * k^2 * (C + 1), roi_h, roi_w)
            h_score_roi = score[0, :, y1:y2, x1:x2]
            h_score_roi = F.reshape(
                h_score_roi, (1, 2 * self.k**2 * (self.C+1), roi_h, roi_w))
            # assembling: (n_batch=1, 2*(C+1), roi_h, roi_w)
            h_score_assm = functions.assemble_2d(h_score_roi, self.k)
            # score map for inside/outside: (n_batch=1, C+1, 2, roi_h, roi_w)
            # ([0,1,2,3, ...] -> [[0,1],[2,3],...]
            h_score_assm = F.reshape(
                h_score_assm, (1, self.C+1, 2, roi_h, roi_w))
            # inside/outside likelihood: (n_batch=1, 2, roi_h, roi_w)
            h_score_inout = h_score_assm[:, cls_id]

            # gt_roi_seg: (n_batch, roi_h, roi_w)
            # get centor of roi's instance number
            # remain only instance number gotten
            gt_roi_seg = t_label_inst_32s[y1:y2, x1:y2]
            roi_center = (roi_h/2,  roi_w/2)
            gt_seg = gt_roi_seg[roi_center]
            gt_roi_seg = np.asarray(gt_roi_seg=gt_seg, dtype=np.int32)

            a_loss_seg = F.softmax_cross_entropy(h_score_inout, gt_roi_seg)
            loss_seg += a_loss_seg

        loss = loss_cls + loss_seg
        return loss

    def _get_rois(self, cls_likelihood, t_label_cls):
        '''git rois whose class score is true and high

        :param np.array cls_likelihood: (n_batch, C+1, height/32, width/32)
            Lable image output from network
        :param np.array t_label_cls: (n_batch, height/32, width/32)
            Grand truth label image about object class.
        '''
        rois = []
        thre = 0.7
        _, height, width = t_label_cls.shape

        for cls_id, a_cls in enumerate(cls_likelihood):
            # ignore background
            if cls_id == 0:
                continue
            # get mask that is above threshold
            # e.g.:[0, 0, 0, cls_id, cls_id, 0, 0, ....]
            mask = (a_cls > thre) * cls_id
            # mask 0 -> -2 (ignore)
            mask[mask == 0] = -2
            # judgement that masked inference class is true or not
            # if true, return roi(cls_id, x1, y1, x2, y2)
            true_class_inds = np.transpose(np.where(mask == t_label_cls))
            for index in true_class_inds:
                x1 = max(0, index[1] - 11/2)
                x2 = min(width, index[1] + 11/2)
                y1 = max(0, index[0] - 11/2)
                y2 = min(height, index[0] + 11/2)
                rois.append((cls_id, x1, y1, x2, y2))

        return rois
