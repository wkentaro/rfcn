import cv2
import dlib
import fcn
import matplotlib.pyplot as plt
import nose.tools
import numpy as np

import rfcn

from test_label2instance_boxes import get_instance_segmentation_data


def test_label_rois():
    img, lbl_cls, lbl_inst = get_instance_segmentation_data()

    rects = []
    dlib.find_candidate_object_locations(img, rects, min_size=500)
    rois = []
    for rect in rects:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        rois.append((x1, y1, x2, y2))
    rois = np.array(rois)

    roi_clss, roi_inst_masks = rfcn.utils.label_rois(
        rois, lbl_inst, lbl_cls, overlap_thresh=0.5)

    n_rois = len(rois)
    nose.tools.assert_equal(len(roi_clss), n_rois)
    nose.tools.assert_equal(len(roi_inst_masks), n_rois)
    np.testing.assert_equal(np.unique(roi_clss), [0, 1])

    viz_imgs = []
    colors = fcn.utils.labelcolormap(21)
    for roi, roi_cls, roi_inst_mask in zip(rois, roi_clss, roi_inst_masks):
        if roi_cls == 0:
            continue
        assert roi_cls > 0
        viz = img.copy()
        x1, y1, x2, y2 = roi
        viz[y1:y2, x1:x2][roi_inst_mask] = 0
        color = (colors[roi_cls][::-1] * 255).astype(np.uint8).tolist()
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)
        viz_imgs.append(viz)

    return fcn.utils.get_tile_image(viz_imgs)


if __name__ == '__main__':
    viz = test_label_rois()
    plt.imshow(viz)
    plt.show()
