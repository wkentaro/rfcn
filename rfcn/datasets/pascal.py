import os.path as osp

import chainer
import numpy as np
import PIL.Image
import scipy

from rfcn.datasets.instance_segmentation import InstanceSegmentationDatasetBase


class PascalInstanceSegmentationDataset(InstanceSegmentationDatasetBase):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        dataset_dir = chainer.dataset.get_dataset_directory(
            'pascal/VOCdevkit/VOC2012')
        imgsets_file = osp.join(
            dataset_dir,
            'ImageSets/Segmentation/{}.txt'.format(data_type))
        self.files = []
        for data_id in open(imgsets_file).readlines():
            data_id = data_id.strip()
            img_file = osp.join(
                dataset_dir, 'JPEGImages/{}.jpg'.format(data_id))
            seg_class_file = osp.join(
                dataset_dir, 'SegmentationClass/{}.png'.format(data_id))
            seg_object_file = osp.join(
                dataset_dir, 'SegmentationObject/{}.png'.format(data_id))
            self.files.append({
                'img': img_file,
                'seg_class': seg_class_file,
                'seg_object': seg_object_file,
            })

    def __len__(self):
        return len(self.files)

    def get_example(self, i):
        """Return data example for instance segmentation of given index.

        Parameters
        ----------
        i: int
            Index of the example.

        Returns
        -------
        datum: numpy.ndarray, (channels, height, width), float32
            Image data.
        label_class: numpy.ndaray, (height, width), int32
            Class label image.
        label_instance: numpy.ndarray, (height, width), int32
            Instance label image.
        """
        data_file = self.files[i]
        # load image
        img_file = data_file['img']
        img = scipy.misc.imread(img_file, mode='RGB')
        datum = self.img_to_datum(img)
        # load class segmentaion gt
        seg_class_file = data_file['seg_class']
        label_class = PIL.Image.open(seg_class_file)
        label_class = np.array(label_class, dtype=np.int32)
        label_class[label_class == 255] = -1
        # load instance segmentation gt
        seg_object_file = data_file['seg_object']
        label_instance = PIL.Image.open(seg_object_file)
        label_instance = np.array(label_instance, dtype=np.int32)
        label_instance[label_instance == 255] = -1
        return datum, label_class, label_instance


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = PascalInstanceSegmentationDataset('val')
    for i in xrange(len(dataset)):
        img_viz = dataset.visualize_example(i)
        plt.imshow(img_viz)
        plt.show()
