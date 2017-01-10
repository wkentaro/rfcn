import collections

import cv2
import fcn
import numpy as np
import PIL.Image


def label2instance_boxes(label_instance, label_class):
    # instance_class is 'Class of the Instance'
    instance_classes = []
    boxes = []
    instances = np.unique(label_instance)
    for inst in instances:
        mask_inst = label_instance == inst
        count = collections.Counter(label_class[mask_inst].tolist())
        instance_class = max(count.items(), key=lambda x: x[1])[0]

        where = np.argwhere(mask_inst)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1

        instance_classes.append(instance_class)
        boxes.append((x1, y1, x2, y2))
    return instance_classes, boxes


def draw_instance_boxes(img, boxes, instance_classes, captions,
                        n_class, bg_class=0, draw_caption=True):
    """Draw labeled rectangles on image.

    Args:
        - img (numpy.ndarray): RGB image.
        - boxes (list of tuple): (x1, y1, x2, y2)

    Returns:
        - img_viz (numpy.ndarray): RGB image.
    """
    if not (len(boxes) == len(instance_classes) == len(captions)):
        raise ValueError

    img_viz = img.copy()
    cmap = fcn.utils.labelcolormap(n_class)

    CV_AA = 16
    for box, inst_class, caption in zip(boxes, instance_classes, captions):
        if inst_class == bg_class:
            continue

        # get color for the label
        color = cmap[inst_class]
        color = (color * 255).tolist()

        x1, y1, x2, y2 = box
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), color[::-1], 2, CV_AA)

        if draw_caption:
            font_scale = 0.4
            ret, baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img_viz, (x1, y2 - ret[1] - baseline),
                          (x1 + ret[0], y2), color[::-1], -1)
            cv2.putText(img_viz, caption, (x1, y2 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        1, CV_AA)

    return img_viz


def mask_to_bbox(mask):
    """Convert mask image to bounding box.

    Parameters
    ----------
    mask: :class:`numpy.ndarray`
        Input mask image.

    Returns
    -------
    box: tuple (x1, y1, x2, y2)
        Bounding box.
    """
    where = np.argwhere(mask)
    (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
    bbox = x1, y1, x2, y2
    return bbox


def label_to_bboxes(label, bg_label=-1):
    """Convert label image to bounding boxes."""
    bboxes = []
    for l in np.unique(label):
        if l == bg_label:
            continue
        mask = label == l
        bbox = mask_to_bbox(mask)
        bboxes.append(bbox)
    return np.array(bboxes)


def resize_image(img, shape):
    height, width = shape[:2]
    img_pil = PIL.Image.fromarray(img)
    img_pil = img_pil.resize((width, height))
    return np.array(img_pil)


def get_bbox_overlap(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    w1, h1 = x12 - x11, y12 - y11
    w2, h2 = x22 - x21, y22 - y21
    intersect = (max(0, min(x12, x22) - max(x11, x21)) *
                 max(0, min(y12, y22) - max(y11, y21)))
    union = w1 * h1 + w2 * h2 - intersect
    return 1.0 * intersect / union
