import os.path as osp

import chainer
import matplotlib.pyplot as plt
import numpy as np


class MovingMNISTDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        dataset_dir = chainer.dataset.get_dataset_directory('pfnet/chainer/mnist')
        if data_type == 'train':
            # (train_size, 784)
            self.npz_data = np.load(osp.join(dataset_dir, 'train.npz'))
        else:
            # (val_size, 784)
            self.npz_data = np.load(osp.join(dataset_dir, 'test.npz'))

    def __len__(self):
        return len(self.npz_data)

    def get_example(self, i):
        # number
        gray = self.npz_data[i]  # (784,)
        gray = gray.reshape((28, 28))

        # move number in space (280, 280)
        data = np.zeros((9, 3, 280, 280), dtype=np.float32)
        for j in xrange(9):
            datum = np.zeros((3, 280, 280), dtype=np.float32)
        # data: (n_batch=9, channels=3, height=280, width=280),
        # masks: (n_batch=9, height=280, width=280)
        return data, masks

    def visualize_example(self, i):
        data = self.get_example(i)
        # visualize here
        img_viz = None
        return img_viz


if __name__ == '__main__':
    dataset = MovingMNIST('val')
    img_viz = dataset.visualize_example(0)
    plt.imshow(img_viz)
    plt.show()
