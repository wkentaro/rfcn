import matplotlib.pyplot as plt

import rfcn

from test_label2instance_boxes import get_instance_segmentation_data


def test_visualize_instance_segmentation():
    img, lbl_cls, lbl_ins = get_instance_segmentation_data()
    class_names = rfcn.datasets.PascalInstanceSegmentationDataset.class_names
    viz = rfcn.utils.visualize_instance_segmentation(
        lbl_ins, lbl_cls, img, class_names)
    return viz


if __name__ == '__main__':
    viz = test_visualize_instance_segmentation()
    plt.imshow(viz)
    plt.show()
