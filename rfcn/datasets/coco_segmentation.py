import os.path as osp

import chainer
import fcn
import numpy as np
import PIL.Image
import PIL.ImageDraw
import skimage.io

import pycocotools
from pycocotools.coco import COCO


class CocoSegmentaionDataset(fcn.datasets.SegmentationDatasetBase):

    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        data_type = data_type + '2014'
        dataset_dir = chainer.dataset.get_dataset_directory('coco')
        ann_file = osp.join(
            dataset_dir, 'annotations/instances_%s.json' % data_type)
        self.coco = COCO(ann_file)
        self.img_fname = osp.join(
            dataset_dir, data_type, 'COCO_%s_{:012}.jpg' % data_type)

        labels = self.coco.loadCats(self.coco.getCatIds())
        max_label = max(labels, key=lambda x: x['id'])['id']
        n_label = max_label + 1
        self.label_names = [None] * n_label
        for label in labels:
            self.label_names[label['id']] = label['name']
        self.label_names[0] = '__background__'

        self.img_ids = self.coco.getImgIds()

    def get_example(self, i):
        img_id = self.img_ids[i]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_fname = self.img_fname.format(img_id)
        img = skimage.io.imread(img_fname)
        datum = self.img_to_datum(img)

        label = self._annotations_to_label(anns, img.shape[0], img.shape[1])

        return datum, label

    @staticmethod
    def _annotations_to_label(anns, height, width):
        label = np.zeros((height, width), dtype=np.int32)
        label.fill(0)
        for ann in anns:
            if 'segmentation' not in ann:
                continue
            if isinstance(ann['segmentation'], list):
                # polygon
                for seg in ann['segmentation']:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask = PIL.Image.fromarray(mask)
                    xy = np.array(seg).reshape((len(seg) / 2, 2))
                    xy = map(tuple, xy)
                    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
                    mask = np.array(mask)
                    label[mask == 1] = ann['category_id']
            else:
                # mask
                if isinstance(ann['segmentation']['counts'], list):
                    rle = pycocotools.mask.frPyObjects(
                        [ann['segmentation']], height, width)
                else:
                    rle = [ann['segmentation']]
                mask = pycocotools.mask.decode(rle)[:, :, 0]
                label[mask == 1] = ann['category_id']
        return label

    def __len__(self):
        return len(self.img_ids)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = CocoSegmentaionDataset('val')
    for i in xrange(len(dataset)):
        img_viz = dataset.visualize_example(i)
        plt.imshow(img_viz)
        plt.show()
