import matplotlib.pyplot as plt

import cv2

import rfcn


if __name__ == '__main__':
    dataset = rfcn.datasets.PascalInstanceSegmentationDataset('val')
    datum, label_class, label_instance = dataset.get_example(0)
    img = dataset.datum_to_img(datum)

    label_instance_fg = label_instance.copy()
    label_instance_fg[label_class == 0] = -1
    boxes = rfcn.utils.label_to_bboxes(label_instance_fg)
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
    plt.imshow(img)
    plt.show()
