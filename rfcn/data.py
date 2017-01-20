import os.path as osp

import chainer
import fcn


def download_faster_rcnn_chainermodel():
    root = chainer.dataset.get_dataset_directory('rfcn')
    path = osp.join(root, 'faster_rcnn_vgg16.npz')
    return fcn.data.cached_download(
        url='https://dl.dropboxusercontent.com/u/2498135/faster-rcnn/VGG16_faster_rcnn_final.model',  # NOQA
        path=path,
        md5='9d96f2e7ea5e7099d410a8ecf6ac67b4',
    )
