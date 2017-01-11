import nose.tools
import numpy as np

from rfcn.datasets import PascalInstanceSegmentationDataset
from rfcn import utils

from test_label2instance_boxes import get_instance_segmentation_data


def test_draw_instance_boxes():
    img, lbl_cls, lbl_inst = get_instance_segmentation_data()
    inst_clss, boxes = utils.label2instance_boxes(
        lbl_inst, lbl_cls, ignore_class=(-1, 0))
    class_names = PascalInstanceSegmentationDataset.class_names
    captions = class_names[inst_clss]
    viz = utils.draw_instance_boxes(
        img, boxes, inst_clss, n_class=len(class_names), captions=captions)

    nose.tools.assert_is_instance(viz, np.ndarray)
    nose.tools.assert_equal(viz.shape, img.shape)
    nose.tools.assert_equal(viz.dtype, np.uint8)

    return viz


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    viz = test_draw_instance_boxes()
    plt.imshow(viz)
    plt.show()
