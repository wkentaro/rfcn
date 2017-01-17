import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import cupy
import fcn
import numpy as np
import sklearn.metrics

from rfcn import functions
from rfcn import utils


class FCIS_SS(chainer.Chain):

    def __init__(self, C, k=7):
        super(FCIS_SS, self).__init__()
        self.C = C
        self.k = k

        # feature extraction:
        self.add_link('conv1_1', L.Convolution2D(3, 64, 3, stride=1, pad=1))
        self.add_link('conv1_2', L.Convolution2D(64, 64, 3, stride=1, pad=1))
        self.add_link('conv2_1', L.Convolution2D(64, 128, 3, stride=1, pad=1))
        self.add_link('conv2_2', L.Convolution2D(128, 128, 3, stride=1, pad=1))
        self.add_link('conv3_1', L.Convolution2D(128, 256, 3, stride=1, pad=1))
        self.add_link('conv3_2', L.Convolution2D(256, 256, 3, stride=1, pad=1))
        self.add_link('conv3_3', L.Convolution2D(256, 256, 3, stride=1, pad=1))
        self.add_link('conv4_1', L.Convolution2D(256, 512, 3, stride=1, pad=1))
        self.add_link('conv4_2', L.Convolution2D(512, 512, 3, stride=1, pad=1))
        self.add_link('conv4_3', L.Convolution2D(512, 512, 3, stride=1, pad=1))

        # translation-aware instance inside/outside score map:
        # out_channel is 2 * k^2 * (C + 1): 2 is inside/outside,
        # k is kernel size, and (C + 1) is object categories and background.
        self.add_link('score_fr', L.Convolution2D(512, 2*k**2*(C+1), ksize=1))

    def _extract_feature(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)  # 1/2

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)  # 1/4

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)  # 1/8

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        # h = F.max_pooling_2d(h, 2, stride=2)

        return h  # 1/8

    def __call__(self, x, lbl_cls, lbl_ins, rois):
        xp = cuda.get_array_module(x.data)
        rois = cuda.to_cpu(rois.data[0])
        lbl_cls = cuda.to_cpu(lbl_cls.data[0])
        lbl_ins = cuda.to_cpu(lbl_ins.data[0])

        if rois.size == 0:
            return

        self.x = x
        self.lbl_cls = lbl_cls
        self.lbl_ins = lbl_ins
        self.rois = rois

        down_scale = 8.0
        h_feature = self._extract_feature(x)
        rois_ns = (rois / down_scale).astype(np.int32)

        # (1, 2*k^2*(C+1), height/down_scale, width/down_scale)
        h_score = self.score_fr(h_feature)  # 1/down_scale
        assert h_score.shape[:2] == (1, 2*self.k**2*(self.C+1))

        roi_clss, roi_segs = utils.label_rois(rois, lbl_ins, lbl_cls)

        loss_cls = Variable(xp.array(0, dtype=np.float32), volatile='auto')
        loss_seg = Variable(xp.array(0, dtype=np.float32), volatile='auto')
        n_loss_cls = 0
        n_loss_seg = 0

        n_rois = len(rois)

        roi_clss_pred = np.zeros((n_rois,), dtype=np.int32)
        cls_scores = np.zeros((n_rois, self.C+1), dtype=np.float32)
        roi_mask_probs = [None] * n_rois

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
            cls_scores[i_roi] = cuda.to_cpu(cls_score.data)[0]
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
                roi_seg = utils.resize_image(roi_seg, (roi_h, roi_w))
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

            roi_mask_prob = F.softmax(roi_score[0])[:, 1, :, :]
            roi_mask_probs[i_roi] = cuda.to_cpu(roi_mask_prob.data)

        if n_loss_cls != 0:
            loss_cls /= n_loss_cls
        if n_loss_seg != 0:
            loss_seg /= n_loss_seg
        loss = loss_cls + loss_seg

        self.loss_cls = float(loss_cls.data)
        self.loss_seg = float(loss_seg.data)
        self.loss = float(loss.data)

        # rois -> label

        # suppress rois with threshold 0.7
        keep = []
        score_argsort = np.argsort(cls_scores.sum(axis=1))
        for i, roi_i in zip(score_argsort, rois[score_argsort]):
            if roi_clss_pred[i] == 0:
                continue
            roi_ns_i = rois_ns[i]
            x1, y1, x2, y2 = roi_ns_i
            roi_h, roi_w = y2 - y1, x2 - x1
            if not (roi_h >= self.k and roi_w >= self.k):
                continue
            if all(utils.get_bbox_overlap(roi_i, rois[j]) < 0.7 for j in keep):
                keep.append(i)
        keep = np.array(keep)

        lbl_cls_pred = np.zeros_like(lbl_cls)
        lbl_ins_pred = np.zeros_like(lbl_ins)
        lbl_ins_pred.fill(-1)

        accumulated = []
        for i in keep:
            if i in accumulated:
                continue

            roi_mask_probs_cum = np.zeros((x.shape[2], x.shape[3], self.C+1),
                                          dtype=np.float64)
            # cls_score_cum = np.zeros((self.C+1,), dtype=np.float64)

            roi_i = rois[i]
            cls_score_i = cls_scores[i]
            # cls_score_cum += cls_score_i
            roi_mask_prob_i = roi_mask_probs[i]
            x1, y1, x2, y2 = roi_i
            roi_mask_prob_i = np.array([utils.resize_image(m, (y2-y1, x2-x1))
                                        for m in roi_mask_prob_i])
            roi_mask_prob_i = roi_mask_prob_i.transpose(1, 2, 0)
            roi_mask_prob_i = 2 * roi_mask_prob_i - 1
            roi_mask_prob_i *= cls_score_i
            roi_mask_probs_cum[y1:y2, x1:x2] += roi_mask_prob_i

            for j in keep:
                roi_j = rois[j]
                if not (0.5 < utils.get_bbox_overlap(roi_i, roi_j) < 1):
                    continue
                assert 0.5 < utils.get_bbox_overlap(roi_i, roi_j) < 0.7
                accumulated.append(j)
                cls_score_j = cls_scores[j]
                # cls_score_cum += cls_score_j
                roi_mask_prob_j = roi_mask_probs[j]
                x1, y1, x2, y2 = roi_j
                roi_mask_prob_j = np.array([
                    utils.resize_image(m, (y2-y1, x2-x1))
                    for m in roi_mask_prob_j])
                roi_mask_prob_j = roi_mask_prob_j.transpose(1, 2, 0)
                roi_mask_prob_j = 2 * roi_mask_prob_j - 1
                roi_mask_prob_j *= cls_score_j
                roi_mask_probs_cum[y1:y2, x1:x2] += roi_mask_prob_j

            # roi_cls = np.argmax(cls_score_cum)
            roi_cls = roi_clss_pred[i]

            if roi_cls != 0:
                # 1/down_scale
                roi_mask_prob = roi_mask_probs_cum[:, :, roi_cls]
                roi_mask = roi_mask_prob > 0
                # 1/1
                roi_mask = utils.resize_image(
                    roi_mask.astype(np.int32),
                    (x.shape[2], x.shape[3])).astype(bool)
                x1, y1, x2, y2 = roi_i
                lbl_cls_pred[roi_mask] = roi_cls
                lbl_ins_pred[roi_mask] = i

        self.roi_clss = roi_clss
        self.roi_clss_pred = roi_clss_pred
        self.lbl_cls_pred = lbl_cls_pred
        self.lbl_ins_pred = lbl_ins_pred

        self.accuracy_cls = sklearn.metrics.accuracy_score(
            roi_clss, roi_clss_pred)
        self.iu_lbl_cls = fcn.utils.label_accuracy_score(
            lbl_cls, self.lbl_cls_pred, self.C+1)[2]
        self.iu_lbl_ins = utils.instance_label_accuracy_score(
            lbl_ins, self.lbl_ins_pred)

        chainer.report({
            'loss': self.loss,
            'loss_cls': self.loss_cls,
            'loss_seg': self.loss_seg,
            'accuracy': self.accuracy_cls,
            'cls_iu': self.iu_lbl_cls,
            'ins_iu': self.iu_lbl_ins,
        }, self)

        return loss
