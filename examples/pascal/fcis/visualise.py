#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path as osp

import chainer
from chainer import cuda
import fcn
import numpy as np
import scipy.misc

import rfcn


def visualize_prediction(model, dataset):
    datum = chainer.cuda.to_cpu(model.x.data[0])
    img = dataset.datum_to_img(datum).copy()
    shape = img.shape

    class_names = dataset.class_names
    # visualize true
    lbl_ins = rfcn.utils.resize_image(model.lbl_ins, shape)
    lbl_cls = rfcn.utils.resize_image(model.lbl_cls, shape)
    img = rfcn.utils.resize_image(img, shape)
    viz_lbl_true = rfcn.utils.visualize_instance_segmentation(
        lbl_ins, lbl_cls, img, class_names)
    # visualize prediction
    lbl_ins_pred = rfcn.utils.resize_image(model.lbl_ins_pred, shape)
    lbl_cls_pred = rfcn.utils.resize_image(model.lbl_cls_pred, shape)
    viz_lbl_pred = rfcn.utils.visualize_instance_segmentation(
        lbl_ins_pred, lbl_cls_pred, img, class_names)
    return fcn.utils.get_tile_image([viz_lbl_true, viz_lbl_pred], (2, 1))


def evaluate(model, iterator, viz_out, device):
    model.train = False
    if not osp.exists(osp.dirname(viz_out)):
        os.makedirs(osp.dirname(viz_out))
    results = []
    vizs = []
    for i, batch in enumerate(iterator):
        in_arrays = [np.asarray(x) for x in zip(*batch)][:3]
        # prediction
        if device >= 0:
            in_arrays = [cuda.to_gpu(x, device=device) for x in in_arrays]
        in_vars = [chainer.Variable(x, volatile=True) for x in in_arrays]
        model(*in_vars)
        results.append((
            model.loss,
            model.loss_cls,
            model.loss_seg,
            #model.accuracy_cls,
            model.iu_lbl_cls,
            model.iu_lbl_ins,
        ))
        if i % 100 == 0:
            # visualization
            viz = visualize_prediction(model, iterator.dataset)
            vizs.append(viz)
    scipy.misc.imsave(viz_out, fcn.utils.get_tile_image(vizs))
    results = np.mean(results, axis=0)
    model.train = True
    return results


def evaluate_once(model, dataset, viz_out, device=0):
    model.train = False
    if not osp.exists(osp.dirname(viz_out)):
        os.makedirs(osp.dirname(viz_out))
    viz_imgs = []
    n_example = 9 if len(dataset) >= 9 else 1
    for i in xrange(n_example):
        batch = [dataset.get_example(i)]
        in_arrays = [np.asarray(x) for x in zip(*batch)][:3]
        if device >= 0:
            in_arrays = [cuda.to_gpu(x, device=device) for x in in_arrays]
        in_vars = [chainer.Variable(x, volatile=True) for x in in_arrays]
        model(*in_vars)
        # visualization
        viz = visualize_prediction(model, dataset)
        viz_imgs.append(viz)
    scipy.misc.imsave(viz_out, fcn.utils.get_tile_image(viz_imgs))
    model.train = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weight')
    parser.add_argument('out')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--single', action='store_true',
                        help='Flag to evaluate only once')
    parser.add_argument('--one-example', action='store_true',
                        help='Flag to use only 1 example')
    parser.add_argument('--parson', action='store_true',
                        help='Flag to use parson dataset')
    args = parser.parse_args()

    weight = args.weight
    out = args.out
    gpu = args.gpu
    single = args.single
    one_example = args.one_example
    person_dataset = args.parson

    if not osp.exists(out):
        os.mkdir(out)
    viz_out = osp.join(out,'pred.png')

    # dataset
    if person_dataset:
        from rfcn.datasets import PascalInstanceSegmentation2ClassRPDataset
        dataset_train = PascalInstanceSegmentation2ClassRPDataset(
            'train', one_example=one_example)
        dataset_val = PascalInstanceSegmentation2ClassRPDataset(
            'val', one_example=one_example)
    else:
        from rfcn.datasets import PascalInstanceSegmentationRPDataset
        dataset_train = PascalInstanceSegmentationRPDataset(
            'train', one_example=one_example)
        dataset_val = PascalInstanceSegmentationRPDataset(
            'val', one_example=one_example)

    iter_train = chainer.iterators.SerialIterator(dataset_train, batch_size=1)
    iter_val = chainer.iterators.SerialIterator(
        dataset_val, batch_size=1, shuffle=False)

    # model
    C = len(dataset_train.class_names) - 1
    model = rfcn.models.CONVFCIS(C=C, k=7)
    chainer.serializers.load_hdf5(weight, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # execute visualize
    if single:
        evaluate_once(model, dataset_val, viz_out, gpu)
    else:
        evaluate(model, iter_val, viz_out, gpu)


if __name__ == '__main__':
    main()
