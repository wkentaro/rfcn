import chainer
import chainer.functions as F
import chainer.links as L

from rfcn.external.faster_rcnn.models.rpn import RPN


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
        self.h_conv1 = h

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv2 = h

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv3 = h

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv4 = h

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        self.h_conv5 = h

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
        h_conv5 = self.trunk(x)

        # TODO(wkentaro): compose im_info from x
        im_info = None
        # TODO(wkentaro): compose gt from t_label_inst
        gt_boxes = None
        rpn_cls_loss, rpn_loss_bbox, rois = self.rpn(
            self.trunk.h_conv4, im_info=im_info, gpu=0, gt_boxes=gt_boxes)

        h_score = self.conv_score(h_conv5)

        for roi in rois:
            # TODO(wkentaro)
            # h_assembled = assemble(h_score[:, :, roi[0]:roi[1], roi[2]:roi[3]])
            scores_cls = []
            for c in xrange(self.C):
                scores_cls.append(np.max(h_assembled[:, 2*c:2*c+1], axis=1))
            pass

        # loss = loss_bbox_objectness + loss_bbox_class + loss_bbox_regression
        # return loss

        pass
