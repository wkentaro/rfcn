import os.path as osp

import cv2
import fcn
import matplotlib.pyplot as plt
import nose.tools
import numpy as np
import PIL.Image
import skimage.color

import rfcn


this_dir = osp.dirname(osp.realpath(__file__))


def test_label2instance_boxes():
    img_file = osp.join(this_dir, '../data/2007_000033_img.jpg')
    lbl_cls_file = osp.join(this_dir, '../data/2007_000033_lbl_cls.png')
    lbl_inst_file = osp.join(this_dir, '../data/2007_000033_lbl_inst.png')
    img = np.array(PIL.Image.open(img_file))
    lbl_cls = np.array(PIL.Image.open(lbl_cls_file))
    lbl_cls[lbl_cls == 255] = -1
    lbl_inst = np.array(PIL.Image.open(lbl_inst_file))
    lbl_inst[lbl_inst == 255] = -1

    inst_clss, bboxes = rfcn.utils.label2instance_boxes(lbl_inst, lbl_cls)

    viz = img.copy()
    colors = fcn.utils.labelcolormap(21)
    for inst_cls, bbox in zip(inst_clss, bboxes):
        x1, y1, x2, y2 = bbox
        color = (colors[inst_cls][::-1] * 255).astype(np.uint8).tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color)

    return viz


if __name__ == '__main__':
    viz = test_label2instance_boxes()
    plt.imshow(viz)
    plt.show()
