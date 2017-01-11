import os.path as osp

import cv2
import fcn
import matplotlib.pyplot as plt
import nose.tools
import numpy as np
import PIL.Image

import rfcn


this_dir = osp.dirname(osp.realpath(__file__))


def test_label2instance_boxes():
    img_file = osp.join(this_dir, '../data/2007_000033_img.jpg')
    lbl_cls_file = osp.join(this_dir, '../data/2007_000033_lbl_cls.png')
    lbl_inst_file = osp.join(this_dir, '../data/2007_000033_lbl_inst.png')
    img = np.array(PIL.Image.open(img_file))
    lbl_cls = np.array(PIL.Image.open(lbl_cls_file)).astype(np.int32)
    lbl_cls[lbl_cls == 255] = -1
    lbl_inst = np.array(PIL.Image.open(lbl_inst_file)).astype(np.int32)
    lbl_inst[lbl_inst == 255] = -1

    inst_clss, bboxes = rfcn.utils.label2instance_boxes(
        lbl_inst, lbl_cls, ignore_class=(-1, 0))

    np.testing.assert_equal(inst_clss, [1, 1, 1])
    nose.tools.assert_equal(len(bboxes[inst_clss]), 3)

    viz = img.copy()
    colors = fcn.utils.labelcolormap(21)
    for inst_cls, bbox in zip(inst_clss, bboxes):
        if inst_cls == 0:
            continue
        x1, y1, x2, y2 = bbox
        color = (colors[inst_cls][::-1] * 255).astype(np.uint8).tolist()
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)
    return viz


if __name__ == '__main__':
    viz = test_label2instance_boxes()
    plt.imshow(viz)
    plt.show()
