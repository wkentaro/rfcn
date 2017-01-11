import os.path as osp

import matplotlib.pyplot as plt
import nose.tools
import numpy as np
import PIL.Image
import skimage.color

import rfcn


this_dir = osp.dirname(osp.realpath(__file__))


def test_resize_image():
    img_file = osp.join(this_dir, '../data/2007_000033_img.jpg')
    lbl_cls_file = osp.join(this_dir, '../data/2007_000033_lbl_cls.png')
    lbl_inst_file = osp.join(this_dir, '../data/2007_000033_lbl_inst.png')
    img = np.array(PIL.Image.open(img_file))
    lbl_cls = np.array(PIL.Image.open(lbl_cls_file)).astype(np.int32)
    lbl_cls[lbl_cls == 255] = -1
    lbl_inst = np.array(PIL.Image.open(lbl_inst_file)).astype(np.int32)
    lbl_inst[lbl_inst == 255] = -1

    height, width = img.shape[:2]
    height_32s, width_32s = height // 32, width // 32
    img_32s = rfcn.utils.resize_image(img, (height_32s, width_32s))
    lbl_cls_32s = rfcn.utils.resize_image(lbl_cls, (height_32s, width_32s))
    lbl_inst_32s = rfcn.utils.resize_image(lbl_inst, (height_32s, width_32s))

    nose.tools.assert_equal(img_32s.shape, (height_32s, width_32s, 3))
    nose.tools.assert_equal(lbl_cls_32s.shape, (height_32s, width_32s))
    nose.tools.assert_equal(lbl_inst_32s.shape, (height_32s, width_32s))

    np.testing.assert_equal(np.unique(lbl_cls), np.unique(lbl_cls_32s))
    np.testing.assert_equal(np.unique(lbl_inst), np.unique(lbl_inst_32s))

    viz_cls = skimage.color.label2rgb(lbl_cls, img)
    viz_cls_32s = skimage.color.label2rgb(lbl_cls_32s, img_32s)
    viz_inst = skimage.color.label2rgb(lbl_inst, img)
    viz_inst_32s = skimage.color.label2rgb(lbl_inst_32s, img_32s)

    return img, viz_cls, viz_inst, img_32s, viz_cls_32s, viz_inst_32s


if __name__ == '__main__':
    img, viz_cls, viz_inst, img_32s, viz_cls_32s, viz_inst_32s = \
        test_resize_image()
    plt.subplot(231)
    plt.imshow(img)
    plt.subplot(232)
    plt.imshow(viz_cls)
    plt.subplot(233)
    plt.imshow(viz_inst)
    plt.subplot(234)
    plt.imshow(img_32s)
    plt.subplot(235)
    plt.imshow(viz_cls_32s)
    plt.subplot(236)
    plt.imshow(viz_inst_32s)
    plt.show()
