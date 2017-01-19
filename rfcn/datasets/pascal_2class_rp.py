import fcn

from rfcn.datasets.pascal_2class import PascalInstanceSegmentation2ClassDataset
from rfcn import utils


class PascalInstanceSegmentation2ClassRPDataset(
        PascalInstanceSegmentation2ClassDataset
        ):

    def __init__(self, data_type, one_example=False,
                 class_name='person', negative_ratio=1.0):
        super(PascalInstanceSegmentation2ClassRPDataset,
              self).__init__(data_type, one_example, class_name)
        self.negative_ratio = negative_ratio

    def get_example(self, i):
        datum, lbl_cls, lbl_ins = \
            super(PascalInstanceSegmentation2ClassRPDataset,
                  self).get_example(i)
        # get rois
        img = self.datum_to_img(datum)
        datum = self.img_to_datum(img)
        rois = utils.get_region_proposals(img)
        keep = utils.nms(rois, 0.9)
        rois = rois[keep]
        roi_clss, _ = utils.label_rois(rois, lbl_ins, lbl_cls)
        samples = utils.get_positive_negative_samples(
            roi_clss != 0, negative_ratio=self.negative_ratio)
        rois = rois[samples]
        return datum, lbl_cls, lbl_ins, rois

    def visualize_example(self, i):
        datum, lbl_cls, lbl_ins, rois = self.get_example(i)
        img = self.datum_to_img(datum)
        # visualize label
        viz_lbl = utils.visualize_instance_segmentation(
            lbl_ins, lbl_cls, img, self.class_names)
        # visualize rois
        roi_clss, _ = utils.label_rois(rois, lbl_ins, lbl_cls)
        viz_all = utils.draw_instance_boxes(
            img, rois, roi_clss, len(self.class_names), bg_class=-1)
        viz_pos = utils.draw_instance_boxes(
            img, rois, roi_clss, len(self.class_names), bg_class=0)
        return fcn.utils.get_tile_image([viz_lbl, viz_all, viz_pos], (1, 3))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = PascalInstanceSegmentation2ClassRPDataset('val')
    for i in xrange(len(dataset)):
        img_viz = dataset.visualize_example(i)
        plt.imshow(img_viz)
        plt.show()
