import fcn

from rfcn import utils


class InstanceSegmentationDatasetBase(fcn.datasets.SegmentationDatasetBase):

    def visualize_example(self, i):
        datum, lbl_cls, lbl_ins = self.get_example(i)
        img = self.datum_to_img(datum)
        viz = utils.visualize_instance_segmentation(
            lbl_ins, lbl_cls, img, self.class_names)
        return viz
