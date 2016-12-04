import chainer
import chainer.links as L

from ..external.faster_rcnn.models.rpn import RPN


class FCIS_VGG16(chainer.Chain):

    """FCIS based on pretrained model of VGG16."""

    def __init__(self, k=3, C=21):
        """Initialize network architecture.

        Parameters
        ----------
        - k (int): kernel size for translation-aware score map. (default: 3)
        - C (int): number of object categories except background. (default: 20)
        """
        super(FCIS, self).__init__()

        # feature extraction: should be copied from VGG16
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
        self.add_link('conv5_1', L.Convolution2D(512, 512, 3, stride=1, pad=1))
        self.add_link('conv5_2', L.Convolution2D(512, 512, 3, stride=1, pad=1))
        self.add_link('conv5_3', L.Convolution2D(512, 512, 3, stride=1, pad=1))

        # rpn:
        self.add_link('rpn', RPN(512, 512, n_anchors=9, feat_stride=16))

        # translation-aware instance inside/outside score map:
        # out_channel is 2 * k^2 * (C + 1): 2 is inside/outside,
        # k is kernel size, and (C + 1) is object categories and background.
        self.add_link('conv_score',
                      L.Convolution2D(512, 2 * k**2 * (C + 1), ksize=1))

    def __call__(self, x, t):
        # TODO(wkentaro): Define computation.
        # loss = loss_bbox_objectness + loss_bbox_class + loss_bbox_regression
        # return loss

        pass
