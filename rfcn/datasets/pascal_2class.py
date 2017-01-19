import os.path as osp
import xml.etree.ElementTree

import numpy as np

from rfcn.datasets.pascal import PascalInstanceSegmentationDataset


class PascalInstanceSegmentation2ClassDataset(
        PascalInstanceSegmentationDataset
        ):

    class_names = np.array([
        'background',
        None,
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, data_type, one_example=False, class_name='person'):
        super(PascalInstanceSegmentation2ClassDataset,
              self).__init__(data_type, one_example)
        self.class_names[1] = class_name
        # update self.files for the target class
        dataset_dir = osp.dirname(osp.dirname(self.files[0]['img']))
        files_new = []
        for file_ in self.files:
            data_id = osp.splitext(osp.basename(file_['img']))[0]
            ann_file = osp.join(
                dataset_dir, 'Annotations/{}.xml'.format(data_id))
            objects = xml.etree.ElementTree.parse(ann_file).findall('object')
            objects = [obj.find('name').text for obj in objects]
            if self.class_names[1] not in objects:
                continue
            files_new.append(file_)
        self.files = files_new

    def get_example(self, i):
        datum, lbl_cls, lbl_ins = \
            super(PascalInstanceSegmentation2ClassDataset, self).get_example(i)
        # remove no need labels
        all_class_names = PascalInstanceSegmentationDataset.class_names
        for lv, ln in enumerate(all_class_names):
            if ln not in self.class_names:
                lbl_ins[lbl_cls == lv] = -1
                lbl_cls[lbl_cls == lv] = -1
        target_class_name = self.class_names[1]
        target_class_value = np.where(all_class_names == target_class_name)[0]
        lbl_cls[lbl_cls == target_class_value] = 1
        return datum, lbl_cls, lbl_ins


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = PascalInstanceSegmentation2ClassDataset('val')
    for i in xrange(len(dataset)):
        img_viz = dataset.visualize_example(i)
        plt.imshow(img_viz)
        plt.show()
