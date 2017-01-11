import os.path as osp

import cv2
import fcn
import matplotlib.pyplot as plt
import nose.tools
import numpy as np
import PIL.Image

import rfcn


this_dir = osp.dirname(osp.realpath(__file__))


def get_instance_segmentation_data():
    img_file = osp.join(this_dir, '../data/2007_000033_img.jpg')
    lbl_cls_file = osp.join(this_dir, '../data/2007_000033_lbl_cls.png')
    lbl_inst_file = osp.join(this_dir, '../data/2007_000033_lbl_inst.png')
    img = np.array(PIL.Image.open(img_file))
    lbl_cls = np.array(PIL.Image.open(lbl_cls_file)).astype(np.int32)
    lbl_cls[lbl_cls == 255] = -1
    lbl_inst = np.array(PIL.Image.open(lbl_inst_file)).astype(np.int32)
    lbl_inst[lbl_inst == 255] = -1
    return img, lbl_cls, lbl_inst


def test_label2instance_boxes():
    img, lbl_cls, lbl_inst = get_instance_segmentation_data()

    inst_clss, bboxes, inst_masks = rfcn.utils.label2instance_boxes(
        lbl_inst, lbl_cls, ignore_class=(-1, 0), return_masks=True)

    n_inst = 3
    height, width = img.shape[:2]
    np.testing.assert_equal(inst_clss, [1, 1, 1])
    nose.tools.assert_equal(len(bboxes[inst_clss]), n_inst)
    nose.tools.assert_equal(inst_masks.shape, (n_inst, height, width))

    viz = img.copy()
    colors = fcn.utils.labelcolormap(21)
    for inst_cls, bbox in zip(inst_clss, bboxes):
        x1, y1, x2, y2 = bbox
        color = (colors[inst_cls][::-1] * 255).astype(np.uint8).tolist()
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)

    viz_masks = inst_masks.astype(np.uint8) * 255
    viz_masks = fcn.utils.get_tile_image(viz_masks, (1, 3))

    return viz, viz_masks


if __name__ == '__main__':
    viz, viz_masks = test_label2instance_boxes()
    plt.subplot(211)
    plt.imshow(viz)
    plt.subplot(212)
    plt.imshow(viz_masks, cmap='gray')
    plt.show()
