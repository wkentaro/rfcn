#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path as osp
import sys

import chainer
from chainer import cuda
import fcn
import numpy as np
import scipy.misc

import rfcn


def visualize_prediction(model, dataset):
    datum = model.datum
    img = dataset.datum_to_img(datum).copy()
    class_names = dataset.class_names
    n_class = len(class_names)
    # visualize true
    lbl_ins = model.lbl_ins
    lbl_cls = model.lbl_cls
    viz_true = rfcn.utils.visualize_instance_segmentation(
        lbl_ins, lbl_cls, img, class_names)
    # visualize prediction
    lbl_ins_pred = model.lbl_ins_pred
    lbl_cls_pred = model.lbl_cls_pred
    viz_pred = rfcn.utils.visualize_instance_segmentation(
        lbl_ins_pred, lbl_cls_pred, img, class_names)
    return fcn.utils.get_tile_image([viz_true, viz_pred], (2, 1))


def evaluate_once(model, dataset, device, viz_out):
    model.train = False
    if not osp.exists(osp.dirname(viz_out)):
        os.makedirs(osp.dirname(viz_out))
    viz_imgs = []
    n_example = 9 if len(dataset) >= 9 else 1
    for i in xrange(n_example):
        batch = [dataset.get_example(i)]
        in_arrays = [np.asarray(x) for x in zip(*batch)]
        if device >= 0:
            in_arrays = [cuda.to_gpu(x, device=device) for x in in_arrays]
        in_vars = [chainer.Variable(x, volatile=True) for x in in_arrays]
        model(*in_vars)
        # visualization
        viz = visualize_prediction(model, dataset)
        viz_imgs.append(viz)
    scipy.misc.imsave(viz_out, fcn.utils.get_tile_image(viz_imgs))
    model.train = True


def evaluate(model, iterator, device, viz_out):
    model.train = False
    if not osp.exists(osp.dirname(viz_out)):
        os.makedirs(osp.dirname(viz_out))
    results = []
    vizs = []
    for i, batch in enumerate(iterator):
        in_arrays = [np.asarray(x) for x in zip(*batch)]
        # prediction
        if device >= 0:
            in_arrays = [cuda.to_gpu(x, device=device) for x in in_arrays]
        in_vars = [chainer.Variable(x, volatile=True) for x in in_arrays]
        model(*in_vars)
        results.append((
            model.loss,
            model.loss_cls,
            model.loss_ins,
            model.iu_cls,
            model.iu_ins,
        ))
        if i % 100 == 0:
            # visualization
            viz = visualize_prediction(model, iterator.dataset)
            vizs.append(viz)
    scipy.misc.imsave(viz_out, fcn.utils.get_tile_image(vizs))
    results = np.mean(results, axis=0)
    model.train = True
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', type=int, default=0, help='default: 0')
    parser.add_argument('--one-example', action='store_true',
                        help='Flag to use only 1 example')
    parser.add_argument('--person-dataset', action='store_true',
                        help='Flag to use person dataset')
    args = parser.parse_args()

    gpu = args.gpu
    out = args.out
    one_example = args.one_example
    person_dataset = args.person_dataset

    if not osp.exists(out):
        os.makedirs(out)

    # 1. dataset

    if person_dataset:
        from rfcn.datasets import PascalInstanceSegmentation2ClassDataset
        dataset_train = PascalInstanceSegmentation2ClassDataset(
            'train', one_example=one_example)
        dataset_val = PascalInstanceSegmentation2ClassDataset(
            'val', one_example=one_example)
    else:
        from rfcn.datasets import PascalInstanceSegmentationDataset
        dataset_train = PascalInstanceSegmentationDataset(
            'train', one_example=one_example)
        dataset_val = PascalInstanceSegmentationDataset(
            'val', one_example=one_example)

    iter_train = chainer.iterators.SerialIterator(dataset_train, batch_size=1)
    iter_val = chainer.iterators.SerialIterator(
        dataset_val, batch_size=1, shuffle=False)

    # 2. model

    fcn_path = '/home/wkentaro/Desktop/FCN32s_model_iter_100000.h5'
    fcn_cls = fcn.models.FCN32s()
    chainer.serializers.load_hdf5(fcn_path, fcn_cls)

    n_class = len(dataset_train.class_names)
    model = rfcn.models.FCNDual(n_class)
    model.train = True
    fcn.utils.copy_chainermodel(fcn_cls, model)

    cuda.get_device(gpu).use()
    if gpu >= 0:
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.Adam(alpha=0.002)
    optimizer.setup(model)

    # training loop

    csv_file = osp.join(out, 'log.csv')
    csv_template = ','.join([
        '{epoch}',
        '{iteration}',
        '{loss}',
        '{loss_cls}',
        '{loss_ins}',
        '{iu_cls}',
        '{iu_ins}',
        '{is_eval}',
    ]) + '\n'
    with open(csv_file, 'w') as f:
        f.write(csv_template.replace('{', '').replace('}', ''))

    max_iter = 10000
    interval_eval = 1000

    epoch = 0
    iteration = 0
    for batch in iter_train:

        iteration += 1
        if iter_train.is_new_epoch:
            epoch += 1

        # training

        in_arrays = [np.asarray(x) for x in zip(*batch)]
        if gpu >= 0:
            in_arrays = [cuda.to_gpu(x, device=gpu) for x in in_arrays]
        in_vars = [chainer.Variable(x) for x in in_arrays]

        model.zerograds()
        loss = model(*in_vars)

        if loss is not None:
            loss.backward()
            optimizer.update()

            with open(csv_file, 'a') as f:
                f.write(csv_template.format(
                    epoch=epoch,
                    iteration=iteration,
                    loss=model.loss,
                    loss_cls=model.loss_cls,
                    loss_ins=model.loss_ins,
                    iu_cls=model.iu_cls,
                    iu_ins=model.iu_ins,
                    is_eval=False,
                ))

        # evaluate

        if iteration % interval_eval == 0:
            viz_out = osp.join(out, 'viz_val', '{}.jpg'.format(iteration))
            results = evaluate(model, iter_val, device=gpu, viz_out=viz_out)
            with open(csv_file, 'a') as f:
                f.write(csv_template.format(
                    epoch=epoch,
                    iteration=iteration,
                    loss=results[0],
                    loss_cls=results[1],
                    loss_ins=results[2],
                    iu_cls=results[3],
                    iu_ins=results[4],
                    is_eval=True,
                ))
            model_out = osp.join(out, '{model}_{iter}.h5'.format(
                model=model.__class__.__name__,
                iter=iteration))
            chainer.serializers.save_hdf5(model_out, model)

        if iteration % 10 == 0:
            viz_out = osp.join(out, 'viz_train', '{}.jpg'.format(iteration))
            evaluate_once(model, dataset_train, device=gpu, viz_out=viz_out)

        # finalize

        if iteration >= max_iter:
            break

    f.close()


if __name__ == '__main__':
    main()
