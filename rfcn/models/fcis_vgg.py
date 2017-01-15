from __future__ import division

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import cupy
import fcn
import numpy as np
import sklearn.metrics

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
        self.h_conv4_8s = h  # 1/8
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv4 = h  # 1/16

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv5 = h  # 1/32


class FCISVGG(chainer.Chain):

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
        super(FCISVGG, self).__init__()
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

    def _propose_regions(self, x, t_label_cls, t_label_inst, h_conv4, gpu):
        # gt_boxes: [[x1, y1, x2, y2, label], ...]
        t_label_cls_data = cuda.to_cpu(t_label_cls.data[0])
        t_label_inst_fg = cuda.to_cpu(t_label_inst.data[0])
        t_label_inst_fg[t_label_cls_data == 0] = -1
        gt_boxes = utils.label_to_bboxes(t_label_inst_fg)
        # propose regions
        # im_info: [[height, width, image_scale], ...]
        _, _, height, width = x.shape
        im_info = np.array([[height, width, 1]], dtype=np.float32)
        # loss_bbox_reg: bbox regression loss
        # rois: (n_rois, 5), [batch_index, x1, y1, x2, y2]
        loss_bbox_cls, loss_bbox_reg, rois = self.rpn(
            self.trunk.h_conv4,  # 1/16
            im_info=im_info,
            gt_boxes=gt_boxes,
            gpu=self.trunk.h_conv4.data.device.id,
        )
        loss_bbox = loss_bbox_cls + loss_bbox_reg
        return loss_bbox, rois, gt_boxes

    def __call__(self, x, t_label_cls, t_label_inst):
        """Forward FCIS with VGG pretrained model.

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
        xp = cuda.get_array_module(x.data)

        # feature extraction
        # ---------------------------------------------------------------------
        self.trunk(x)
        h_conv4 = self.trunk.h_conv4  # 1/16
        h_conv5 = self.trunk.h_conv5  # 1/32
        # ---------------------------------------------------------------------

        # region proposals
        # ---------------------------------------------------------------------
        assert x.shape[0] == 1  # only supports 1 size batch
        loss_bbox_reg, rois, _ = self._propose_regions(
            x, t_label_cls, t_label_inst, h_conv4, gpu=h_conv4.data.device.id)
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
        _, _, height_32s, width_32s = h_conv5.shape
        shape_32s = (height_32s, width_32s, 3)
        t_label_cls_32s = utils.resize_image(
            cuda.to_cpu(t_label_cls.data), shape_32s)
        t_label_inst_32s = utils.resize_image(
            cuda.to_cpu(t_label_inst.data), shape_32s)
        n_rois = 0
        for roi in rois:
            x1, y1, x2, y2 = [r // 32 for r in roi[1:]]
            roi_h, roi_w = y2 - y1, x2 - x1

            # create gt_roi_cls & gt_roi_seg
            roi_label_inst = t_label_inst_32s[y1:y2, x1:x2]
            max_overlap = 0
            gt_roi_cls = None
            gt_roi_seg = None
            for lbl_inst in np.unique(roi_label_inst):
                gt_mask = t_label_inst_32s == lbl_inst
                roi_mask = np.zeros_like(gt_mask)
                roi_mask[y1:y2, x1:x2] = True
                intersect = gt_mask[y1:y2, x1:x2].sum()
                union = np.bitwise_or(gt_mask, roi_mask).sum()
                overlap = 1. * intersect / union
                if overlap > max_overlap:
                    max_overlap = overlap
                    unique, count = np.unique(
                        t_label_cls_32s[gt_mask],
                        return_counts=True)
                    gt_roi_cls = unique[np.argmax(count)]
                    # 0: outside, 1: inside
                    gt_roi_seg = t_label_cls_32s == gt_roi_cls
                    gt_roi_seg = np.bitwise_and(gt_mask[y1:y2, x1:x2],
                                                gt_roi_seg[y1:y2, x1:x2])
            if max_overlap < 0.5:
                continue
            n_rois += 1
            # gt_roi_cls: (n_batch=1,)
            gt_roi_cls = gt_roi_cls.reshape(1,)
            if xp == cupy:
                gt_roi_cls = cuda.to_gpu(gt_roi_cls)
            gt_roi_cls = gt_roi_cls.astype(np.int32)
            gt_roi_cls = chainer.Variable(gt_roi_cls, volatile='auto')
            # gt_roi_seg: (n_batch=1, roi_h, roi_w)
            gt_roi_seg = gt_roi_seg.reshape(1, roi_h, roi_w)
            if xp == cupy:
                gt_roi_seg = cuda.to_gpu(gt_roi_seg)
            gt_roi_seg = gt_roi_seg.astype(np.int32)
            gt_roi_seg = chainer.Variable(gt_roi_seg, volatile='auto')

            # score map in ROI: (n_batch=1, 2 * k^2 * (C + 1), roi_h, roi_w)
            h_score_roi = h_score[0, :, y1:y2, x1:x2]
            h_score_roi = F.reshape(
                h_score_roi, (1, 2 * self.k**2 * (self.C+1), roi_h, roi_w))
            # assembling: (n_batch=1, 2*(C+1), roi_h, roi_w)
            h_score_assm = functions.assemble_2d(h_score_roi, self.k)
            # score map for inside/outside: (n_batch=1, C+1, 2, roi_h, roi_w)
            h_score_assm = F.reshape(
                h_score_assm, (1, self.C+1, 2, roi_h, roi_w))
            # class likelihood: (n_batch=1, C+1, roi_h, roi_w)
            h_cls_likelihood = F.max(h_score_assm, axis=2)
            # (n_batch=1, C+1)
            h_cls_likelihood = F.sum(h_cls_likelihood, axis=(2, 3))

            a_loss_cls = F.softmax_cross_entropy(h_cls_likelihood, gt_roi_cls)
            loss_cls += a_loss_cls
            # inside/outside likelihood: (n_batch=1, 2, roi_h, roi_w)
            h_cls_id = F.argmax(h_cls_likelihood, axis=1)
            h_score_inout = h_score_assm[:, int(h_cls_id.data[0])]
            a_loss_seg = F.softmax_cross_entropy(h_score_inout, gt_roi_seg)
            loss_seg += a_loss_seg
        # ---------------------------------------------------------------------

        # loss_cls: mask classification loss
        # loss_seg: mask segmentation loss
        loss = loss_bbox_reg
        if n_rois != 0:
            loss_cls /= n_rois
            loss_seg /= n_rois
            loss += (loss_cls + loss_seg)

        chainer.report({'loss': loss}, self)

        return loss


class FCISVGG_RP(FCISVGG):

    def __call__(self, x, t_label_cls, t_label_inst):
        self.x = x

        # feature extraction
        self.trunk(x)
        h_conv4 = self.trunk.h_conv4  # 1/16

        # region proposals
        loss, rois, gt_boxes = self._propose_regions(
            x, t_label_cls, t_label_inst, h_conv4, gpu=h_conv4.data.device.id)
        self.rois = rois
        self.gt_boxes = gt_boxes

        # compute mean iu
        iu_scores = []
        for gt in gt_boxes:
            overlaps = [utils.get_bbox_overlap(gt[:4], roi[1:])
                        for roi in rois]
            iu_scores.append(max(overlaps))
        mean_iu = np.mean(iu_scores)

        chainer.report({'loss': loss, 'iu': mean_iu}, self)
        return loss


class FCISVGG_SS(chainer.Chain):

    def __init__(self, C, k=7):
        super(FCISVGG_SS, self).__init__()
        self.C = C
        self.k = k

        # feature extraction:
        self.add_link('trunk', VGG16Trunk())

        # translation-aware instance inside/outside score map:
        # out_channel is 2 * k^2 * (C + 1): 2 is inside/outside,
        # k is kernel size, and (C + 1) is object categories and background.
        self.add_link('score_fr', L.Convolution2D(512, 2*k**2*(C+1), ksize=1))

    def __call__(self, x, lbl_cls, lbl_ins, rois):
        xp = cuda.get_array_module(x.data)
        rois = cuda.to_cpu(rois.data[0])
        lbl_cls = cuda.to_cpu(lbl_cls.data[0])
        lbl_ins = cuda.to_cpu(lbl_ins.data[0])

        self.x = x
        self.lbl_cls = lbl_cls
        self.lbl_ins = lbl_ins
        self.rois = rois

        self.trunk(x)

        down_scale = 8.0
        h_feature = self.trunk.h_conv4_8s

        # (1, 2*k^2*(C+1), height/down_scale, width/down_scale)
        h_score = self.score_fr(h_feature)  # 1/down_scale
        assert h_score.shape[:2] == (1, 2*self.k**2*(self.C+1))

        shape_ns = h_feature.shape[2:4]
        lbl_cls_ns = utils.resize_image(lbl_cls, shape_ns)
        lbl_ins_ns = utils.resize_image(lbl_ins, shape_ns)
        rois_ns = (rois / down_scale).astype(np.int32)

        try:
            roi_clss, roi_segs = utils.label_rois(
                rois_ns, lbl_ins_ns, lbl_cls_ns)
        except Exception:
            return Variable(xp.array(0, dtype=np.float32), volatile='auto')

        loss_cls = Variable(xp.array(0, dtype=np.float32), volatile='auto')
        loss_seg = Variable(xp.array(0, dtype=np.float32), volatile='auto')
        n_loss_cls = 0
        n_loss_seg = 0

        n_rois = len(rois)

        roi_clss_pred = np.zeros((n_rois,), dtype=np.int32)
        cls_scores = np.zeros((n_rois,), dtype=np.float32)
        roi_masks_pred = [None] * n_rois

        for i_roi in xrange(n_rois):
            roi_ns = rois_ns[i_roi]
            roi_cls = roi_clss[i_roi]
            roi_seg = roi_segs[i_roi]

            roi_cls_var = xp.array([roi_cls], dtype=np.int32)
            roi_cls_var = Variable(roi_cls_var, volatile='auto')

            x1, y1, x2, y2 = roi_ns
            roi_h = y2 - y1
            roi_w = x2 - x1

            if not (roi_h >= self.k and roi_w >= self.k):
                continue
            assert roi_h * roi_w > 0

            roi_score = h_score[:, :, y1:y2, x1:x2]
            assert roi_score.shape == (1, 2*self.k**2*(self.C+1), roi_h, roi_w)

            roi_score = functions.assemble_2d(roi_score, self.k)
            assert roi_score.shape == (1, 2*(self.C+1), roi_h, roi_w)

            roi_score = F.reshape(roi_score, (1, self.C+1, 2, roi_h, roi_w))

            cls_score = F.max(roi_score, axis=2)
            assert cls_score.shape == (1, self.C+1, roi_h, roi_w)
            cls_score = F.sum(cls_score, axis=(2, 3))
            cls_score /= (roi_h * roi_w)
            cls_scores[i_roi] = float(F.argmax(cls_score, axis=1).data[0])
            assert cls_score.shape == (1, self.C+1)

            a_loss_cls = F.softmax_cross_entropy(cls_score, roi_cls_var)
            loss_cls += a_loss_cls
            n_loss_cls += 1

            roi_cls_pred = F.argmax(cls_score, axis=1)
            roi_cls_pred = int(roi_cls_pred.data[0])
            roi_clss_pred[i_roi] = roi_cls_pred

            roi_score_io = roi_score[:, roi_cls, :, :, :]
            assert roi_score_io.shape == (1, 2, roi_h, roi_w)

            if roi_cls != 0:
                roi_seg = roi_seg.astype(np.int32)
                roi_seg = roi_seg[np.newaxis, :, :]
                if xp == cupy:
                    roi_seg = cuda.to_gpu(roi_seg, device=x.data.device)
                roi_seg = Variable(roi_seg, volatile='auto')
                a_loss_seg = F.softmax_cross_entropy(roi_score_io, roi_seg)
                loss_seg += a_loss_seg
                n_loss_seg += 1

            roi_score_io = cuda.to_cpu(roi_score_io.data)[0]
            roi_seg_pred = np.argmax(roi_score_io, axis=0)
            roi_seg_pred = roi_seg_pred.astype(bool)

            if roi_cls_pred != 0:
                roi_masks_pred[i_roi] = roi_seg_pred

        if n_loss_cls != 0:
            loss_cls /= n_loss_cls
        if n_loss_seg != 0:
            loss_seg /= n_loss_seg
        loss = loss_cls + loss_seg

        self.roi_clss = roi_clss
        self.roi_clss_pred = roi_clss_pred

        accuracy = sklearn.metrics.accuracy_score(roi_clss, roi_clss_pred)

        keep = roi_clss_pred != 0
        rois = rois[keep]
        roi_clss_pred = roi_clss_pred[keep]
        cls_scores = cls_scores[keep]
        roi_masks_pred = np.array(roi_masks_pred)[keep]
        lbl_ins_pred = np.zeros_like(lbl_ins)
        lbl_cls_pred = np.zeros_like(lbl_cls)
        rois_exists = []
        for i_roi in np.argsort(cls_scores):
            roi = rois[i_roi]
            obj_id = i_roi
            if rois_exists:
                overlaps = [(utils.get_bbox_overlap(roi, r), i)
                            for r, i in rois_exists]
                max_overlap, id_max_overlap = max(overlaps)
                if max_overlap > 0.3:
                    obj_id = id_max_overlap
            rois_exists.append((roi, obj_id))
            x1, y1, x2, y2 = roi
            roi_mask_pred = roi_masks_pred[i_roi]
            roi_mask_pred = utils.resize_image(
                roi_mask_pred.astype(np.uint8), (y2-y1, x2-x1))
            roi_mask_pred = roi_mask_pred == 1
            roi_cls = roi_clss_pred[i_roi]
            lbl_ins_pred[y1:y2, x1:x2][roi_mask_pred] = obj_id
            lbl_cls_pred[y1:y2, x1:x2][roi_mask_pred] = roi_cls

        self.lbl_cls_pred = lbl_cls_pred
        self.lbl_ins_pred = lbl_ins_pred

        cls_iu = fcn.utils.label_accuracy_score(
            lbl_cls, self.lbl_cls_pred, self.C+1)[2]
        ins_iu = utils.instance_label_accuracy_score(
            lbl_ins, self.lbl_ins_pred)

        chainer.report({
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_seg': loss_seg,
            'accuracy': accuracy,
            'cls_iu': cls_iu,
            'ins_iu': ins_iu,
        }, self)

        return loss
