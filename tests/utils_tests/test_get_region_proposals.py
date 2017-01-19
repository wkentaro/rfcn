import cv2
import fcn
import nose.tools
import numpy as np

import rfcn

from test_label2instance_boxes import get_instance_segmentation_data


def test_get_region_proposals():
    img = get_instance_segmentation_data()[0]

    rois = rfcn.utils.get_region_proposals(img)

    nose.tools.assert_is_instance(rois, np.ndarray)
    nose.tools.assert_true(rois.shape[0] > 0)
    nose.tools.assert_equal(rois.shape[1], 4)
    nose.tools.assert_equal(rois.dtype, np.int32)

    viz = img.copy()
    colors = fcn.utils.labelcolormap()
    for i_roi, roi in enumerate(rois):
        x1, y1, x2, y2 = roi
        color = (colors[i_roi % len(colors)][::-1] * 255).astype(int).tolist()
        cv2.rectangle(viz, (x1, y1), (x2, y2), color)

    return img, viz


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img, viz = test_get_region_proposals()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(viz)
    plt.show()
