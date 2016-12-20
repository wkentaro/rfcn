from __future__ import division

import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import numpy as np

from rfcn.external.faster_rcnn.models.rpn import RPN
from rfcn import functions
from rfcn import utils


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


class FCISVGG16(chainer.Chain):

    """FCIS based on pretrained model of VGG16."""

    def __init__(self, C, k=7):
        """Initialize network architecture.

        Parameters
        ----------
        C: int
            number of object categories except background.
        k: int (default: 7)
            kernel size for translation-aware score map.
        """
        super(FCISVGG16, self).__init__()
        self.C = C
        self.k = k

        # feature extraction:
        self.add_link('trunk', VGG16Trunk())

        # rpn:
        self.add_link('rpn', RPN(512, 512, n_anchors=9, feat_stride=16))

        # translation-aware instance inside/outside score map:
        # out_channel is 2 * k^2 * (C + 1): 2 is inside/outside,
        # k is kernel size, and (C + 1) is object categories and background.
        self.add_link('conv_score',
                      L.Convolution2D(512, 2 * k**2 * (C + 1), ksize=1))

    def __call__(self, x, t_label_cls, t_label_inst):
        """Forward FCIS with VGG16 pretrained model.

        This model has three losses:
        - rpn_cls_loss:
            a softmax classification loss over C + 1 categories.
        - label_fg_loss:
            a softmax segmentation loss 1 over the foreground mask
            of the ground-truth category (if roi is positive).
        - rpn_loss_bbox:
            a bbox regression loss (if roi is positive).

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

        # feature extraction
        # ---------------------------------------------------------------------
        # (n_batch, 3, height/32, width/32)
        h_conv5 = self.trunk(x)
        # ---------------------------------------------------------------------

        # region proposals
        # ---------------------------------------------------------------------
        # im_info: [[height, width, image_scale], ...]
        n_batch, _, height_32, width_32 = h_conv5.data.shape
        im_info = np.array([[height_32, width_32, 1]], dtype=np.float32)
        im_info = np.repeat(im_info, n_batch, axis=0)

        # gt_boxes: [[x1, y1, x2, y2, label], ...]
        gt_boxes = np.zeros((n_batch, 4), dtype=np.float32)
        t_label_cls_data = chainer.cuda.to_cpu(t_label_cls.data)
        t_label_inst_data = chainer.cuda.to_cpu(t_label_inst.data)
        for i in xrange(n_batch):
            for lbl_inst in np.unique(t_label_inst_data[i]):
                mask = t_label_inst_data[i] == lbl_inst
                x1, y1, x2, y2 = utils.mask_to_bbox(mask)
                gt_boxes[i][0:3] = x1, y1, x2, y2

                # FCIS does not care about bbox class
                gt_boxes[i][4] = 0
                # lbl_cls_count = collections.Counter(
                #     t_label_cls_data[i][mask].flatten())
                # lbl_cls = max(lbl_cls_count.items())[0]
                # gt_boxes[i][4] = lbl_cls

        # propose regions
        # loss_bbox_reg: bbox regression loss
        # rois: (n_rois, 5), [batch_index, x1, y1, x2, y2]
        _, loss_bbox_reg, rois = self.rpn(
            self.trunk.h_conv4,
            im_info=im_info,
            gt_boxes=gt_boxes,
            gpu=self.gpu,
        )

        # ---------------------------------------------------------------------

        # position sensitive convolution
        # ---------------------------------------------------------------------
        # (n_batch, n_channels=2*k^2*(C+1), height/32, width/32)
        h_score = self.conv_score(h_conv5)  # 1/32
        # ---------------------------------------------------------------------

        # operation for each ROI
        # ---------------------------------------------------------------------
        loss_cls = chainer.Variable(xp.array(0, dtype=np.float32),
                                    volatile='auto')
        loss_seg = chainer.Variable(xp.array(0, dtype=np.float32),
                                    volatile='auto')
        for roi in rois:
            batch_ind, x1, y1, x2, y2 = roi
            roi_h = y2 - y1
            roi_w = x2 - x2

            # create gt_roi_cls & gt_roi_seg
            roi_label_inst = t_label_inst_data[y1:y2, x1:x2]
            max_overlap = 0
            gt_roi_cls = None
            gt_roi_seg = None
            for lbl_inst in np.unique(roi_label_inst):
                gt_mask = t_label_inst_data[batch_ind] == lbl_inst
                roi_mask = np.zeros_like(gt_mask)
                roi_mask[y1:y2, x1:x2] = True
                intersect = gt_mask[y1:y2, x1:x2].sum()
                union = np.bitwise_or(gt_mask, roi_mask).sum()
                overlap = 1. * intersect / union
                if overlap > max_overlap:
                    max_overlap = overlap
                    unique, count = np.unique(
                        t_label_cls_data[batch_ind][gt_mask],
                        return_counts=True)
                    gt_roi_cls = unique[np.argmax(count)]
                    # 0: outside, 1: inside
                    gt_label_seg = t_label_cls_data[batch_ind] == gt_roi_cls
                    gt_label_seg = np.bitwise_and(gt_mask[y1:y2, x1:x2],
                                                  gt_label_seg[y1:y2, x1:x2])
            if max_overlap < 0.5:
                continue
            # gt_roi_cls: (n_batch=1, 1)
            gt_roi_cls = gt_roi_cls.reshape(1, 1)
            if xp == cupy:
                gt_roi_cls = chainer.to_gpu(gt_roi_cls)
            gt_roi_cls = gt_roi_cls.astype(np.int32)
            gt_roi_cls = chainer.Variable(gt_roi_cls, volatile='auto')
            assert gt_roi_cls.shape == (1, 1)
            # gt_roi_seg: (n_batch=1, 1, roi_h, roi_w)
            gt_roi_seg = gt_roi_seg.reshape(1, 1, roi_h, roi_w)
            if xp == cupy:
                gt_roi_seg = chainer.to_gpu(gt_roi_seg)
            gt_roi_seg = gt_roi_seg.astype(np.int32)
            gt_roi_seg = chainer.Variable(gt_roi_seg, volatile='auto')
            assert gt_roi_seg.shape == (1, 1, roi_h, roi_w)

            # score map in ROI: (n_batch=1, 2 * k^2 * (C + 1), roi_h, roi_w)
            h_score_roi = h_score[batch_ind, :, y1:y2, x1:x2]
            h_score_roi = F.reshape(
                h_score_roi, (1, 2 * self.k**2 * (self.C+1), roi_h, roi_w))
            # assembling: (n_batch=1, 2*(C+1), roi_h, roi_w)
            h_score_assm = functions.assemble_2d(h_score_roi, self.k)
            # score map for inside/outside: (n_batch=1, C+1, 2, roi_h, roi_w)
            h_score_assm = F.reshape(
                h_score_assm, (n_batch, self.C+1, 2, roi_h, roi_w))
            # class likelihood: (n_batch=1, C+1, roi_h, roi_w)
            h_cls_likelihood = F.max(h_score_assm, axis=2)
            # (n_batch=1, C+1)
            h_cls_likelihood = F.sum(h_cls_likelihood, axis=(2, 3))

            a_loss_cls = F.softmax_cross_entropy(
                h_cls_likelihood, gt_roi_cls)
            loss_cls += a_loss_cls
            # inside/outside likelihood: (n_batch=1, 2, roi_h, roi_w)
            h_cls_id = F.argmax(h_cls_likelihood, axis=1)
            h_score_inout = h_score_assm[:, h_cls_id]
            a_loss_seg = F.softmax_cross_entropy(h_score_inout, gt_roi_seg)
            loss_seg += a_loss_seg
        # ---------------------------------------------------------------------

        # loss_cls: mask classification loss
        # loss_seg: mask segmentation loss
        loss = loss_bbox_reg + loss_cls + loss_seg
        return loss
