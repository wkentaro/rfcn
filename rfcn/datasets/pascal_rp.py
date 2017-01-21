import fcn
import numpy as np

from rfcn.external.faster_rcnn.faster_rcnn.proposal_target_layer \
    import ProposalTargetLayer
from rfcn.datasets.pascal import PascalInstanceSegmentationDataset
from rfcn import utils


class PascalInstanceSegmentationRPDataset(PascalInstanceSegmentationDataset):

    """Pascal VOC2012 instance segmentation dataset with region proposals"""

    def __init__(self, data_type, one_example=False, negative_ratio=1.0):
        super(PascalInstanceSegmentationRPDataset,
              self).__init__(data_type, one_example)
        n_class = len(self.class_names)
        self.propose_targets = ProposalTargetLayer(num_classes=n_class)
        self.propose_targets.FG_THRESH = 0.8
        self.negative_ratio = negative_ratio

    def get_example(self, i):
        datum, lbl_cls, lbl_ins = \
            super(PascalInstanceSegmentationRPDataset, self).get_example(i)
        # get rois
        img = self.datum_to_img(datum)
        datum = self.img_to_datum(img)
        gt_boxes = utils.label_to_bboxes(lbl_ins)
        roi_clss, _ = utils.label_rois(
            gt_boxes, lbl_ins, lbl_cls, overlap_thresh=0.9)
        gt_boxes = np.hstack((gt_boxes, roi_clss[:, np.newaxis]))
        rois = utils.get_region_proposals(img)
        rois = np.hstack((np.zeros((len(rois), 1)), rois))
        rois, roi_clss, _, _, _ = self.propose_targets(rois, gt_boxes)
        rois = rois[:, 1:].astype(np.int32)
        rois = np.vstack((rois, gt_boxes[:, :4]))
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
    dataset = PascalInstanceSegmentationRPDataset('val')
    for i in xrange(len(dataset)):
        img_viz = dataset.visualize_example(i)
        plt.imshow(img_viz)
        plt.show()
