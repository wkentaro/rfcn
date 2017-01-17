import cv2
import fcn
import matplotlib.pyplot as plt
import nose.tools
import numpy as np

import rfcn

from test_label2instance_boxes import get_instance_segmentation_data


def test_get_positive_negative_samples():
    img, lbl_cls, lbl_inst = get_instance_segmentation_data()

    rois = rfcn.utils.get_region_proposals(img, min_size=500)

    roi_clss, _ = rfcn.utils.label_rois(rois, lbl_inst, lbl_cls)

    is_sample = rfcn.utils.get_positive_negative_samples(roi_clss != 0)
    rois = rois[is_sample]
    roi_clss = roi_clss[is_sample]

    nose.tools.assert_is_instance(is_sample, np.ndarray)
    nose.tools.assert_equal(is_sample.dtype, np.int64)
    nose.tools.assert_equal((roi_clss != 0).sum() * 2, len(is_sample))

    viz_all = img.copy()
    viz_pos = img.copy()
    colors = fcn.utils.labelcolormap(21)
    for roi, roi_cls in zip(rois, roi_clss):
        x1, y1, x2, y2 = roi
        color = (colors[roi_cls][::-1] * 255).astype(np.uint8).tolist()
        cv2.rectangle(viz_all, (x1, y1), (x2, y2), color)
        if roi_cls != 0:
            cv2.rectangle(viz_pos, (x1, y1), (x2, y2), color)

    return fcn.utils.get_tile_image([viz_all, viz_pos], (1, 2))


if __name__ == '__main__':
    viz = test_get_positive_negative_samples()
    plt.imshow(viz)
    plt.show()
