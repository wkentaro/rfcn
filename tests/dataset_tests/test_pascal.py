import cv2
import matplotlib.pyplot as plt
import fcn
import numpy as np

from rfcn.datasets.pascal import PascalInstanceSegmentationDataset
from rfcn import utils


def test_pascal():
    dataset = PascalInstanceSegmentationDataset('train', rp=False)
    viz = dataset.visualize_example(0)

    dataset = PascalInstanceSegmentationDataset('train', rp=True)
    datum, lbl_cls, lbl_ins, rois = dataset.get_example(0)
    img = dataset.datum_to_img(datum)
    rois_clss, _ = utils.label_rois(rois, lbl_ins, lbl_cls)
    colors = fcn.utils.labelcolormap(len(dataset.class_names))

    viz_all = img.copy()
    viz_pos = img.copy()
    for roi, roi_cls in zip(rois, rois_clss):
        x1, y1, x2, y2 = roi
        color = (colors[roi_cls][::-1] * 255).astype(np.uint8).tolist()
        cv2.rectangle(viz_all, (x1, y1), (x2, y2), color)
        if roi_cls != 0:
            cv2.rectangle(viz_pos, (x1, y1), (x2, y2), color)

    return viz, viz_all, viz_pos


if __name__ == '__main__':
    viz, viz_all, viz_pos = test_pascal()
    plt.subplot(131)
    plt.imshow(viz)
    plt.subplot(132)
    plt.imshow(viz_all)
    plt.subplot(133)
    plt.imshow(viz_pos)
    plt.show()
