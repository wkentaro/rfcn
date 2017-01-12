#!/usr/bin/env python

import os.path as osp

import chainer

import fcn.data
import fcn.utils


def main():
    dataset_dir = chainer.dataset.get_dataset_directory('coco')

    path = osp.join(dataset_dir, 'train2014.zip')
    fcn.data.cached_download(
        url='http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
        path=path,
        md5='0da8c0bd3d6becc4dcb32757491aca88',
    )
    fcn.util.extract_file(path, to_directory=dataset_dir)

    path = osp.join(dataset_dir, 'val2014.zip')
    fcn.data.cached_download(
        url='http://msvocds.blob.core.windows.net/coco2014/val2014.zip',
        path=path,
        md5='a3d79f5ed8d289b7a7554ce06a5782b3',
    )
    fcn.util.extract_file(path, to_directory=dataset_dir)

    path = osp.join(dataset_dir, 'instances_train-val2014.zip')
    fcn.data.cached_download(
        url='http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip',  # NOQA
        path=path,
        md5='59582776b8dd745d649cd249ada5acf7',
    )
    fcn.util.extract_file(path, to_directory=dataset_dir)


if __name__ == '__main__':
    main()
