#!/usr/bin/env python

import argparse
import os
import os.path as osp

import chainer
from chainer import cuda
import fcn
import numpy as np
import scipy.misc

import rfcn


from train_ss import evaluate_once
from train_ss import visualize_prediction


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
            model.loss_rpn,
            model.loss_cls,
            model.loss_seg,
            model.accuracy_cls,
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

    vgg_path = fcn.data.download_vgg16_chainermodel()
    vgg = fcn.models.VGG16()
    chainer.serializers.load_hdf5(vgg_path, vgg)

    C = len(dataset_train.class_names) - 1
    model = rfcn.models.FCIS(C=C, k=7)
    model.train = True
    fcn.utils.copy_chainermodel(vgg, model)

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
        '{loss_rpn}',
        '{loss_cls}',
        '{loss_seg}',
        '{accuracy_cls}',
        '{iu_lbl_cls}',
        '{iu_lbl_ins}',
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
                    loss_rpn=model.loss_rpn,
                    loss_cls=model.loss_cls,
                    loss_seg=model.loss_seg,
                    accuracy_cls=model.accuracy_cls,
                    iu_lbl_cls=model.iu_lbl_cls,
                    iu_lbl_ins=model.iu_lbl_ins,
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
                    loss_rpn=results[1],
                    loss_cls=results[2],
                    loss_seg=results[3],
                    accuracy_cls=results[4],
                    iu_lbl_cls=results[5],
                    iu_lbl_ins=results[6],
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
