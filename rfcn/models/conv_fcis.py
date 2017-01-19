# !/usr/bin/python
import PIL

from chainer import cuda
from chainer import Variable
import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import fcn
import numpy as np

from rfcn import functions
from rfcn import utils
from rfcn.models.fcis_vgg import VGG16Trunk

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

        # TODO(mukuact): need to load pre-trained weight
        self.trunk(x)
        h_conv4 = self.trunk.h_conv4  # 1/16
        # (b_batch, 2 * k^2 * (C + 1), height/32, width/32)
        score = self.conv_score(h_conv4)

        sc_shape = score.data.shape

        _, _, height_32s, width_32s = sc_shape
        # grand truth
        t_label_inst_data = chainer.cuda.to_cpu(t_label_inst.data)
        t_label_inst_pil = PIL.Image.fromarray(t_label_inst_data[0])
        t_label_inst_32s = t_label_inst_pil.resize((width_32s, height_32s))
        t_label_inst_32s = np.asarray(t_label_inst_32s)
        lbl_ins = t_label_inst_32s
        t_label_inst_32s = t_label_inst_32s[np.newaxis, ...]
        if xp == cupy:
            t_label_inst_32s = cuda.to_gpu(t_label_inst_32s, device=x.data.device)
        t_label_inst_32s = Variable(t_label_inst_32s, volatile='auto')

        t_label_cls_data = chainer.cuda.to_cpu(t_label_cls.data)
        t_label_cls_pil = PIL.Image.fromarray(t_label_cls_data[0])
        t_label_cls_32s = t_label_cls_pil.resize((width_32s, height_32s))
        t_label_cls_32s = np.asarray(t_label_cls_32s)
        lbl_cls = t_label_cls_32s
        t_label_cls_32s = t_label_cls_32s[np.newaxis, ...]
        if xp == cupy:
            t_label_cls_32s= cuda.to_gpu(t_label_cls_32s, device=x.data.device)
        t_label_cls_32s= Variable(t_label_cls_32s, volatile='auto')

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
            a_cls_likelihood = self.conv_2(h)
            cls_likelihood.append(a_cls_likelihood)

        # (n_batch, C+1, height/32, width/32)
        cls_likelihood = F.concat(cls_likelihood, axis=1)
        assert cls_likelihood.shape == (1, C+1, height_32s, width_32s)
        # roi's grand truth is assumed as center of roi
        # grand truth is seemed to have a same resolution as the score map
        loss_cls = F.softmax_cross_entropy(cls_likelihood, t_label_cls_32s)

        # get rois according to cls score
        # class is true and score > 0.7
        rois, cls_scores = self._get_rois(cls_likelihood, t_label_cls_32s)
        loss_seg = chainer.Variable(xp.array(0, dtype=np.float32),
                                    volatile='auto')
        n_loss_seg = 0

        nrois = len(rois)
        roi_mask_probs = [None] * nrois
        for i, roi in enumerate(rois):
            cls_id, x1, y1, x2, y2 = roi
            roi_h = y2 - y1
            roi_w = x2 - x1
            if roi_h == 0 or roi_w == 0:
                continue

            # score map in ROI: (n_batch=1, 2 * k^2 * (C + 1), roi_h, roi_w)
            h_score_roi = score[0, :, y1:y2, x1:x2]
            h_score_roi = F.reshape(
                h_score_roi, (1, 2 * k**2 * (C+1), roi_h, roi_w))
            # assembling: (n_batch=1, 2*(C+1), roi_h, roi_w)
            # 2nd axis is ordered like 1st pos of each cat, 2nd pos of each cat, 3rd
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
        loss = loss_cls + loss_seg

        if nrois == 0:
            chainer.report({
                'loss': loss,
                'loss_cls': loss_cls,
                'loss_seg': loss_seg,
            }, self)
            return loss
        else:
            lbl_ins_pred, lbl_cls_pred = utils.roi_scores_to_label(
                    (height_32s, width_32s), np.array(rois)[:, 1:], cls_scores, roi_mask_probs,
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

    def _get_rois(self, cls_likelihood, t_label_cls):
        """Get rois whose class score is true and high.

        :param np.array cls_likelihood: (n_batch, C+1, height/32, width/32)
            Lable image output from network
        :param np.array t_label_cls: (n_batch, height/32, width/32)
            Grand truth label image about object class.
        """
        f = self.f
        rois = []
        cls_scores = []
        thre = 0.7
        _, height, width = t_label_cls.shape

        cls_likelihood = cuda.to_cpu(cls_likelihood.data)[0]
        t_label_cls = cuda.to_cpu(t_label_cls.data)[0]

        for cls_id, a_cls in enumerate(cls_likelihood):
            # ignore background
            if cls_id == 0:
                continue
            # get mask that is above threshold
            # e.g.:[0, 0, 0, cls_id, cls_id, 0, 0, ....]
            assert a_cls.shape == t_label_cls.shape
            mask = (a_cls > thre) * cls_id
            # mask 0 -> -2 (ignore)
            mask[mask == 0] = -2
            # judgement that masked inference class is true or not
            # if true, return roi(cls_id, x1, y1, x2, y2)
            true_class_inds = np.transpose(np.where(mask == t_label_cls))
            for index in true_class_inds:
                x1 = max(0, index[1] - f/2)
                x2 = min(width, index[1] + f/2)
                y1 = max(0, index[0] - f/2)
                y2 = min(height, index[0] + f/2)
                rois.append((cls_id, x1, y1, x2, y2))
                cls_scores.append(cls_likelihood[:, index[0], index[1]])

        cls_scores = np.array(cls_scores, np.float32)
        return rois, cls_scores

