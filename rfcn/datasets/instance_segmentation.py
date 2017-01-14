import fcn
import numpy as np
import skimage.color

import rfcn


class InstanceSegmentationDatasetBase(fcn.datasets.SegmentationDatasetBase):

    def visualize_example(self, i):
        datum, label_class, label_instance = self.get_example(i)
        img = self.datum_to_img(datum)

        label_instance[label_class == 0] = -1
        img_viz = skimage.color.label2rgb(label_instance, img)
        img_viz = (img_viz * 255).astype(np.uint8)

        instance_classes, boxes = rfcn.utils.label2instance_boxes(
            label_instance, label_class, ignore_class=(-1, 0))
        img_viz = rfcn.utils.draw_instance_boxes(
            img_viz, boxes, instance_classes,
            n_class=len(self.class_names),
            captions=self.class_names[instance_classes])

        return img_viz
