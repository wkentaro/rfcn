import cv2
import fcn
import nose.tools
import numpy as np
import skimage.color
import skimage.data
import skimage.segmentation

from rfcn import utils


def test_label_to_bboxes():
    img = skimage.data.astronaut()
    label = skimage.segmentation.slic(img)
    label[label % 2 == 0] = -1
    bboxes = utils.label_to_bboxes(label, ignore_label=(-1, 0))

    nose.tools.assert_is_instance(bboxes, np.ndarray)

    # test number of bboxes
    nose.tools.assert_equal(len(bboxes), 28)

    def get_bbox_size(bbox):
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        return height * width

    # test bbox size
    nose.tools.assert_true(all(get_bbox_size(bbox) > 0 for bbox in bboxes))

    # test bbox shape
    height, width = img.shape[:2]
    nose.tools.assert_true(all(0 <= bbox[0] and bbox[2] <= width
                               for bbox in bboxes))
    nose.tools.assert_true(all(0 <= bbox[1] and bbox[3] <= height
                               for bbox in bboxes))

    # visualize bboxes
    viz = skimage.color.label2rgb(label, img)
    colors = fcn.utils.labelcolormap()[1:]
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = (colors[i] * 255).astype(int).tolist()
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)
    return viz


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    viz = test_label_to_bboxes()
    plt.imshow(viz)
    plt.show()
