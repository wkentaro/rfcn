# !/usr/bin/python
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import cupy
import fcn
import numpy as np

from rfcn import functions
from rfcn.models.fcis_vgg import VGG16Trunk
from rfcn import utils


class CONV_FCIS(chainer.Chain):
    def __init__(self, C, k=5):
        f = 15
        super(CONV_FCIS, self).__init__(
            trunk=VGG16Trunk(),
            conv_score=L.Convolution2D(512, 2 * k**2 * (C + 1), ksize=1),

            conv_1=L.Convolution2D(2*(k**2), k, f, pad=(f-1)/2),
            conv_2=L.Convolution2D(k, 1, f, pad=(f-1)/2),

        )

        self.f = f
        self.ksize = k
        self.C = C

    def __call__(self, x, t_label_cls, t_label_inst):
        """Forward FCIS with Convolution.

        Parameters
        ----------
        x.data: (n_batch, 3, height, width)
            Input image.
        t_label_cls: (n_batch, height, width)
            Label image about object class.
        t_label_inst: (n_batch, height, width)
            Label image about object instances.
        """
        xp = chainer.cuda.get_array_module(x.data)
        C = self.C
        k = self.ksize

        self.trunk(x)
        h_conv4 = self.trunk.h_conv4  # 1/16

        # (b_batch, 2 * k^2 * (C + 1), height/32, width/32)
        score = self.conv_score(h_conv4)

        _, _, height_32s, width_32s = score.data.shape
        # grand truth
        lbl_ins = cuda.to_cpu(t_label_inst.data[0])
        t_label_inst_32s = utils.resize_image(lbl_ins, (width_32s, height_32s))
        t_label_inst_32s = t_label_inst_32s[np.newaxis, ...]
        assert t_label_inst_32s.shape == (1, width_32s, height_32s)

        lbl_cls = cuda.to_cpu(t_label_cls.data[0])
        t_label_cls_32s = utils.resize_image(lbl_cls, (width_32s, height_32s))
        t_label_cls_32s = t_label_cls_32s[np.newaxis, ...]
        assert t_label_cls_32s.shape == (1, width_32s, height_32s)

        if xp == cupy:
            t_label_cls_32s = cuda.to_gpu(
                t_label_cls_32s, device=x.data.device)
            t_label_inst_32s = cuda.to_gpu(
                t_label_inst_32s, device=x.data.device)

        # get 2*(k**2) score map for 'ic'th category
        # and apply conv func to them
        # iks:[0, 1, 2(C+1), 2(C+1)+1, 2*2(C+1), 2*2(C+1)+1, 2*3(C+1), ...]
        iks_fore = np.arange(0, 2*(C+1)*(k**2), 2*(C+1))
        iks_back = np.arange(0, 2*(C+1)*(k**2), 2*(C+1)) + 1
        iks = np.ravel(np.dstack((iks_fore, iks_back)))
        cls_likelihood = []
        for ic in xrange(C+1):
            # applying same convolution filter to the maps of each category
            a_cls_score = F.get_item(score, np.s_[:, iks+ic*2])
            h = F.relu(self.conv_1(a_cls_score))
            a_cls_likelihood = F.sigmoid(self.conv_2(h))
            cls_likelihood.append(a_cls_likelihood)

        # (n_batch, C+1, height/32, width/32)
        cls_likelihood = F.concat(cls_likelihood, axis=1)
        assert cls_likelihood.shape == (1, C+1, height_32s, width_32s)

        # get rois according to cls score
        # class is true and score > 0.7
        rois, cls_scores = self._get_rois(cls_likelihood)
        # n_batch = 1
        rois = rois[0]
        cls_scores = cls_scores[0]
        norm_rois = np.array(rois)[:, 1:]

        n_loss_seg = 0
        loss_seg = chainer.Variable(xp.array(0, dtype=np.float32),
                                    volatile='auto')
        nrois = len(rois)
        roi_mask_probs = [None] * nrois
        for i, roi in enumerate(rois):
            cls_id, x1, y1, x2, y2 = roi
            roi_h = y2 - y1
            roi_w = x2 - x1
            if roi_h == 0 or roi_w == 0:
                continue

            # score map in ROI: (n_batch=1, 2 * k^2 * (C + 1), roi_h, roi_w)
            # 2nd axis is ordered like 1st pos of each cat, 2nd pos of each cat
            h_score_roi = score[0, :, y1:y2, x1:x2]
            h_score_roi = F.reshape(
                h_score_roi, (1, 2 * k**2 * (C+1), roi_h, roi_w))
            # assembling: (n_batch=1, 2*(C+1), roi_h, roi_w)
            h_score_assm = functions.assemble_2d(h_score_roi, k)
            # score map for inside/outside: (n_batch=1, C+1, 2, roi_h, roi_w)
            # ([0,1,2,3, ...] -> [[0,1],[2,3],...]
            h_score_assm = F.reshape(
                h_score_assm, (1, self.C+1, 2, roi_h, roi_w))
            # inside/outside likelihood: (n_batch=1, 2, roi_h, roi_w)
            h_score_inout = h_score_assm[:, cls_id]
            assert h_score_inout.shape == (1, 2, roi_h, roi_w)

            roi_mask_prob = F.softmax(h_score_assm[0])[:, 1, :, :]
            roi_mask_probs[i] = cuda.to_cpu(roi_mask_prob.data)

            # gt_roi_seg: (n_batch, roi_h, roi_w)
            # TODO(mukuact) change grand truth
            # get centor of roi's instance number
            # remain only instance number gotten
            gt_roi_seg = t_label_inst_32s[:, y1:y2, x1:x2]
            gt_roi_seg = gt_roi_seg.data
            assert gt_roi_seg.shape == (1, roi_h, roi_w)
            roi_center = (0, roi_h/2,  roi_w/2)
            gt_seg = gt_roi_seg[roi_center]
            gt_roi_seg = xp.asarray(gt_roi_seg == gt_seg, dtype=np.int32)

            a_loss_seg = F.softmax_cross_entropy(h_score_inout, gt_roi_seg)
            loss_seg += a_loss_seg
            n_loss_seg += 1

        if n_loss_seg != 0:
            loss_seg /= n_loss_seg

        t_cls_likelihood = self._rois_cls_gt(
                rois, t_label_cls_32s, t_label_inst_32s)
        assert t_cls_likelihood.shape == cls_likelihood.shape

        # ignore fields that dont belong to any rois
        mask = t_cls_likelihood == 0
        t_cls_likelihood[mask] = cuda.to_cpu(cls_likelihood.data)[mask]
        # calc classification loss
        loss_cls = F.mean_squared_error(cls_likelihood, t_cls_likelihood)

        loss = loss_cls + loss_seg

        # store some variables
        self.x = x
        self.lbl_cls = lbl_cls
        self.lbl_ins = lbl_ins
        self.rois = norm_rois
        self.loss = cuda.to_cpu(loss.data)
        self.loss_cls = cuda.to_cpu(loss_cls.data)
        self.loss_seg = cuda.to_cpu(loss_seg.data)

        if nrois == 0:
            chainer.report({
                'loss': loss,
                'loss_cls': loss_cls,
                'loss_seg': loss_seg,
            }, self)
            return loss
        else:
            lbl_ins_pred, lbl_cls_pred = utils.roi_scores_to_label(
                    (height_32s, width_32s),
                    norm_rois,
                    cls_scores,
                    roi_mask_probs,
                    1, self.ksize, self.C)

            self.lbl_cls_pred = lbl_cls_pred
            self.lbl_ins_pred = lbl_ins_pred

            self.iu_lbl_cls = fcn.utils.label_accuracy_score(
                lbl_cls, lbl_cls_pred, C+1)[2]
            self.iu_lbl_ins = utils.instance_label_accuracy_score(
                lbl_ins, lbl_ins_pred)

            chainer.report({
                'loss': loss,
                'loss_cls': loss_cls,
                'loss_seg': loss_seg,
                'cls_iu': self.iu_lbl_cls,
                'ins_iu': self.iu_lbl_ins,
            }, self)

            return loss

    def _get_rois(self, cls_likelihood):
        """Get rois whose class score is true and high.

        Parameters
        ----------
        cls_likelihood: chainer.Variable (n_batch, C+1, height/32, width/32)
            Lable image output from network
        t_label_cls: np.ndarray (n_batch, height/32, width/32)
            Grand truth label image about object class.

        Returns
        -------
        rois: list of taple
            [[(cls_id, x1, y1, x2, y2),(),(),]]
            causion::1st axis has batch size.
        cls_scores: list of np.ndarray [(n_roi, C+1),]
        """
        cls_likelihood = cuda.to_cpu(cls_likelihood.data)

        f = self.f
        n_batch, _, height, width = cls_likelihood.shape
        rois = []
        cls_scores = []

        for a_batch in cls_likelihood:
            roi = []
            cls_score = []
            assert a_batch.shape[0] == self.C+1

            # high_prob_index: [[cls,y,x],...]
            # ignore background
            high_prob_index = np.argsort(a_batch[1:], axis=None)[::-1]
            high_prob_index = np.unravel_index(
                    high_prob_index, a_batch[1:].shape)
            high_prob_index = np.stack(high_prob_index, axis=-1)

            for index in high_prob_index[:300]:
                if index[1] < f/2 or index[1] > width - f/2:
                    continue
                if index[2] < f/2 or index[2] > height - f/2:
                    continue
                x1 = max(0, index[2] - f/2)
                x2 = min(width, index[2] + f/2)
                y1 = max(0, index[1] - f/2)
                y2 = min(height, index[1] + f/2)
                # +1 for slide cls index because of ignoring background
                roi.append((index[0]+1, x1, y1, x2, y2))
                cls_score.append(a_batch[:, index[1], index[2]])
            rois.append(roi)
            cls_scores.append(np.array(cls_score, np.float32))

        assert cls_scores[0].shape == (len(rois[0]), self.C+1)

        return rois, cls_scores

    def _rois_cls_gt(self, rois, lbl_cls, lbl_ins):
        """calc grand truth of cls_likelihood

        cls_likelihood should donate the ratio of class in the roi.
        It seems hard to calc that for all pixels, so this method
        calcs them for rois and fills other field with -1 for
        ignoring

        Parameters
        ----------
        rois: np.ndarray (C+1, 4)
            (x1, y1, x2, y2)
            These values are for the same resolution as lable images.
        lbl_cls: (height, width)
            class label grand truth
        lbl_ins: (height, width)
            instance label grand truth

        Returns
        -------
        t_cls_likelihood: np.ndarray (1, C+1, height, width)
        """
        import ipdb;ipdb.set_trace()
        roi_values = []
        f = self.f
        height, width = lbl_cls.shape
        for roi in rois:
            x1, y1, x2, y2 = roi
            roi_w = x2 - x1
            roi_h = y2 - y1

            roi_center = (y1+f/2, x1+f/2)

            # get a high count inst in the roi
            roi_lbl_ins = lbl_ins[y1:y2, x1:x2]
            roi_lbl_cls = lbl_cls[y1:y2, x1:x2]
            (values, count) = np.unique(roi_lbl_ins, return_counts=True)
            argsort_count = np.argsort(count)[::-1]
            high_inst = values[argsort_count]
            high_count = count[argsort_count]

            class_ratio = {}
            for i, count in zip(high_inst, high_count):
                # get a class of the high count inst in the roi
                mask = (roi_lbl_ins == i)
                cls_v, cls_c = np.unique(roi_lbl_cls[mask])
                i_cls = cls_v[cls_c[-1]]
                if i_cls in class_ratio:
                    continue
                else:
                    ratio_inst = float(count) / (roi_w * roi_h)
                    class_ratio[i_cls] = ratio_inst

            roi_values.append((roi_center, class_ratio))

        # write the result to image
        t_cls_likelihood = np.zeros((self.C+1, height, width))
        for a_roi in roi_values:
            c_y, c_x = a_roi[0]
            class_ratio = a_roi[1]
            for i_cls in class_ratio:
                t_cls_likelihood[i_cls, c_y, c_x] = class_ratio[i_cls]

        return t_cls_likelihood[np.newaxis, ...]
