import math

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import fcn
import numpy as np

from rfcn import utils


class FCNDual(chainer.Chain):

    def __init__(self, n_class=21, n_proposals=100):
        self.n_class = n_class
        self.n_proposals = n_proposals
        super(self.__class__, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=100),
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

            # class segmentation
            fc6=L.Convolution2D(512, 4096, 7, stride=1, pad=0),
            fc7=L.Convolution2D(4096, 4096, 1, stride=1, pad=0),
            score_fr=L.Convolution2D(4096, self.n_class, 1, stride=1, pad=0),
            upscore=L.Deconvolution2D(self.n_class, self.n_class, 64,
                                      stride=32, pad=0),

            # instance segmentation
            fc6_ins=L.Convolution2D(512, 4096, 7, stride=1),
            fc7_ins=L.Convolution2D(4096, 4096, 1, stride=1),
            score_fr_ins=L.Convolution2D(4096, self.n_proposals, 1,
                                         stride=1, pad=0),
            upscore_ins=L.Deconvolution2D(
                self.n_proposals, 2 * self.n_proposals, 64, stride=32, pad=0),
        )
        self.train = False

    def __call__(self, x, lbl_cls, lbl_ins):
        assert x.shape[0] == 1

        device = x.data.device.id

        self.datum = cuda.to_cpu(x.data[0])
        self.lbl_cls = cuda.to_cpu(lbl_cls.data[0])
        self.lbl_ins = cuda.to_cpu(lbl_ins.data[0])

        # conv1
        h = F.relu(self.conv1_1(x))
        conv1_1 = h
        h = F.relu(self.conv1_2(conv1_1))
        conv1_2 = h
        h = F.max_pooling_2d(conv1_2, 2, stride=2, pad=0)
        pool1 = h  # 1/2

        # conv2
        h = F.relu(self.conv2_1(pool1))
        conv2_1 = h
        h = F.relu(self.conv2_2(conv2_1))
        conv2_2 = h
        h = F.max_pooling_2d(conv2_2, 2, stride=2, pad=0)
        pool2 = h  # 1/4

        # conv3
        h = F.relu(self.conv3_1(pool2))
        conv3_1 = h
        h = F.relu(self.conv3_2(conv3_1))
        conv3_2 = h
        h = F.relu(self.conv3_3(conv3_2))
        conv3_3 = h
        h = F.max_pooling_2d(conv3_3, 2, stride=2, pad=0)
        pool3 = h  # 1/8

        # conv4
        h = F.relu(self.conv4_1(pool3))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool4 = h  # 1/16

        # conv5
        h = F.relu(self.conv5_1(pool4))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool5 = h  # 1/32

        # class segmentation --------------------------------------------------

        # fc6
        h = F.relu(self.fc6(pool5))
        h = F.dropout(h, ratio=.5, train=self.train)
        fc6 = h  # 1/32

        # fc7
        h = F.relu(self.fc7(fc6))
        h = F.dropout(h, ratio=.5, train=self.train)
        fc7 = h  # 1/32

        # score_fr
        h = self.score_fr(fc7)
        score_fr = h  # 1/32

        # upscore
        h = self.upscore(score_fr)
        upscore = h  # 1/1

        # score
        h = upscore[:, :, 19:19+x.data.shape[2], 19:19+x.data.shape[3]]
        score = h  # 1/1

        loss_cls = F.softmax_cross_entropy(score, lbl_cls)

        lbl_cls_pred = F.argmax(score, axis=1)
        self.lbl_cls_pred = cuda.to_cpu(lbl_cls_pred.data[0])

        # instance segmentation -----------------------------------------------

        # fc6
        h = F.relu(self.fc6(pool5))
        h = F.dropout(h, ratio=.5, train=self.train)
        fc6 = h  # 1/32

        # fc7
        h = F.relu(self.fc7(fc6))
        h = F.dropout(h, ratio=.5, train=self.train)
        fc7 = h  # 1/32

        # score_fr_ins
        h = self.score_fr_ins(fc7)
        score_fr_ins = h  # 1/32

        # upscore_ins
        h = self.upscore_ins(score_fr_ins)
        upscore_ins = h  # 1/1

        # score_ins
        h = upscore_ins[:, :, 19:19+x.data.shape[2], 19:19+x.data.shape[3]]
        score_ins = h  # 1/1

        n_batch, channels, height, width = score_ins.shape
        assert n_batch == 1
        assert channels == 2 * self.n_proposals
        # 2 is for fore/background
        score_ins = F.reshape(
            score_ins, (n_batch, self.n_proposals, 2, height, width))
        score_mask = F.max(score_ins, axis=2)
        score_mask = F.sum(score_mask, axis=(2, 3))

        fg_mask = self.lbl_cls_pred != 0
        lbl_ins_pred = np.zeros_like(self.lbl_ins)
        lbl_ins_pred.fill(-1)

        loss_ins = np.array(0, dtype=np.float32)
        loss_ins = cuda.to_gpu(loss_ins, device=device)
        loss_ins = chainer.Variable(loss_ins, volatile='auto')

        lbl_ins_mask = np.bitwise_and(self.lbl_ins != -1, self.lbl_cls != 0)
        ins_ids = np.unique(self.lbl_ins[lbl_ins_mask])
        ins_ids = ins_ids[ins_ids != -1]
        ins_ids_found = []
        score_mask_data = cuda.to_cpu(score_mask.data)[0]
        for i_proposal in np.argsort(score_mask_data)[::-1]:
            score_fb = score_ins[:, i_proposal, :, :, :]

            losses_proposal_i = []

            # if all background

            mask_all_bg = np.zeros((1, height, width), dtype=np.int32)
            if device >= 0:
                mask_all_bg = cuda.to_gpu(mask_all_bg, device=device)
            mask_all_bg = chainer.Variable(mask_all_bg, volatile='auto')

            loss_proposal_bg = F.softmax_cross_entropy(score_fb, mask_all_bg)
            losses_proposal_i.append(loss_proposal_bg)

            # if an instance

            for ins_id in ins_ids:
                mask_ins_i = self.lbl_ins == ins_id
                mask_ins_i = mask_ins_i.astype(np.int32)
                mask_ins_i = mask_ins_i[np.newaxis, :, :]
                if device >= 0:
                    mask_ins_i = cuda.to_gpu(mask_ins_i, device=device)
                mask_ins_i = chainer.Variable(mask_ins_i, volatile='auto')

                loss_proposal_i = F.softmax_cross_entropy(score_fb, mask_ins_i)
                losses_proposal_i.append(loss_proposal_i)

            losses_proposal_i = F.vstack(losses_proposal_i)
            loss_ins += F.min(losses_proposal_i)

            ins_id_index = int(F.argmin(losses_proposal_i).data) - 1
            if ins_id_index >= 0:
                ins_ids_found.append(ins_ids[ins_id_index])

            ###############
            # reconstruct #
            ###############
            mask_proposal = F.argmax(score_fb, axis=1)
            mask_proposal = cuda.to_cpu(mask_proposal.data)[0]
            mask_proposal = mask_proposal.astype(bool)
            assert mask_proposal.shape == (height, width)

            lbl_ins_pred[mask_proposal] = i_proposal

            if (lbl_ins_pred[fg_mask] == -1).sum() == 0:
                break

        lbl_ins_pred[self.lbl_cls == 0] = -1
        self.lbl_ins_pred = lbl_ins_pred

        for ins_id in ins_ids:
            if ins_id in ins_ids_found:
                continue
            mask_ins_i = self.lbl_ins == ins_id
            mask_ins_i = mask_ins_i.astype(np.int32)
            mask_ins_i = mask_ins_i[np.newaxis, :, :]
            if device >= 0:
                mask_ins_i = cuda.to_gpu(mask_ins_i, device=device)
            mask_ins_i = chainer.Variable(mask_ins_i, volatile='auto')

            losses_ins_i = []
            for i_proposal in xrange(self.n_proposals):
                score_fb = score_ins[:, i_proposal, :, :, :]
                loss_ins_i = F.softmax_cross_entropy(
                    score_fb, mask_ins_i)
                losses_ins_i.append(loss_ins_i)

            losses_ins_i = F.vstack(losses_ins_i)
            loss_ins_i = F.min(losses_ins_i)
            loss_ins += loss_ins_i

        loss = loss_cls + loss_ins

        self.loss_cls = cuda.to_cpu(loss_cls.data)
        self.loss_ins = cuda.to_cpu(loss_ins.data)
        self.loss = cuda.to_cpu(loss.data)

        if math.isnan(loss.data):
            raise ValueError('loss value is nan')

        iu_cls = fcn.utils.label_accuracy_score(
            self.lbl_cls, self.lbl_cls_pred, self.n_class)[2]
        iu_ins = utils.instance_label_accuracy_score(
            self.lbl_ins, self.lbl_ins_pred)

        self.iu_cls = iu_cls
        self.iu_ins = iu_ins

        return loss
