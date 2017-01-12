import cv2
import nose.tools
import numpy as np

from rfcn import utils


def test_mask_to_bbox():
    mask = np.zeros((100, 100), dtype=bool)
    mask[38:50, 52:71] = True
    bbox = utils.mask_to_bbox(mask)

    nose.tools.assert_equal(bbox, (52, 38, 71, 50))

    viz = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
    viz = viz.astype(np.uint8) * 255
    x1, y1, x2, y2 = bbox
    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0))

    return viz


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    viz = test_mask_to_bbox()
    plt.imshow(viz)
    plt.show()
