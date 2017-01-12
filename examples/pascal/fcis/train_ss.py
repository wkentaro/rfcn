#!/usr/bin/env python

from __future__ import print_function

import argparse
import datetime
import os
import os.path as osp
import sys

import chainer
from chainer.training import extensions
import cv2
import fcn
import numpy as np
import yaml

import rfcn


def get_trainer(
        dataset_train,
        dataset_val,
        optimizer,
        gpu,
        max_iter,
        out=None,
        resume=None,
        interval_log=10,
        interval_eval=1000,
        ):

    if isinstance(gpu, list):
        gpus = gpu
    else:
        gpus = [gpu]

    if out is None:
        if resume:
            out = osp.dirname(resume)
        else:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            out = osp.join(this_dir, 'logs', timestamp)
    if not resume and osp.exists(out):
        print('Result dir already exists: {}'.format(osp.abspath(out)),
              file=sys.stderr)
        quit(1)
    os.makedirs(out)
    print('Writing result to: {}'.format(osp.abspath(out)))

    # dump parameters
    param_file = osp.join(out, 'param.yaml')
    params = {
        'dataset': {
            'train': {
                'name': dataset_train.__class__.__name__,
                'size': len(dataset_train),
            },
            'val': {
                'name': dataset_val.__class__.__name__,
                'size': len(dataset_val),
            },
        },
        'model': 'FCISVGG_SS',
        'optimizer': {
            'name': optimizer.__class__.__name__,
            'params': optimizer.__dict__,
        },
        'resume': resume,
    }
    yaml.safe_dump(params, open(param_file, 'w'), default_flow_style=False)
    print('>' * 20 + ' Parameters ' + '>' * 20)
    yaml.safe_dump(params, sys.stdout, default_flow_style=False)
    print('<' * 20 + ' Parameters ' + '<' * 20)

    # 1. dataset
    if len(gpus) > 1:
        iter_train = chainer.iterators.MultiprocessIterator(
            dataset_train, batch_size=len(gpus), shared_mem=10000000)
    else:
        iter_train = chainer.iterators.SerialIterator(
            dataset_train, batch_size=1)
    iter_val = chainer.iterators.SerialIterator(
        dataset_val, batch_size=1, repeat=False, shuffle=False)

    # 2. model
    vgg_path = fcn.data.download_vgg16_chainermodel()
    vgg = fcn.models.VGG16()
    chainer.serializers.load_hdf5(vgg_path, vgg)

    n_classes = len(dataset_train.class_names) - 1
    model = rfcn.models.FCISVGG_SS(C=n_classes, k=3)
    model.train = True
    fcn.utils.copy_chainermodel(vgg, model.trunk)

    if len(gpus) > 1 or gpus[0] >= 0:
        chainer.cuda.get_device(gpus[0]).use()
    if len(gpus) == 1 and gpus[0] >= 0:
        model.to_gpu()

    # 3. optimizer
    optimizer.setup(model)

    # 4. trainer
    if len(gpus) > 1:
        devices = {'main': gpus[0]}
        for gpu in gpus[1:]:
            devices['gpu{}'.format(gpu)] = gpu
        updater = chainer.training.ParallelUpdater(
            iter_train, optimizer, devices=devices)
    else:
        updater = chainer.training.StandardUpdater(
            iter_train, optimizer, device=gpus[0])
    trainer = chainer.training.Trainer(
        updater, (max_iter, 'iteration'), out=out)

    trainer.extend(
        fcn.training.extensions.TestModeEvaluator(
            iter_val, model, device=gpus[0]),
        trigger=(interval_eval, 'iteration'),
        invoke_before_training=False,
    )

    def visualize_ss(target):
        datum = chainer.cuda.to_cpu(target.x.data[0])
        img = dataset_val.datum_to_img(datum).copy()
        lbl_cls = chainer.cuda.to_cpu(target.lbl_cls.data[0])
        lbl_ins = chainer.cuda.to_cpu(target.lbl_ins.data[0])
        rois = target.rois
        colors = fcn.utils.labelcolormap(21)
        # visualize true roi clss
        roi_clss = target.roi_clss
        viz_true = img.copy()
        for roi, roi_cls in zip(rois, roi_clss):
            if roi_cls == 0:
                continue
            color = (colors[roi_cls][::-1] * 255).astype(np.uint8).tolist()
            x1, y1, x2, y2 = roi
            cv2.rectangle(viz_true, (x1, y1), (x2, y2), color)
        viz_pred = img.copy()
        roi_clss_pred = target.roi_clss_pred
        for roi, roi_cls in zip(rois, roi_clss_pred):
            if roi_cls == 0:
                continue
            color = (colors[roi_cls][::-1] * 255).astype(np.uint8).tolist()
            x1, y1, x2, y2 = roi
            cv2.rectangle(viz_pred, (x1, y1), (x2, y2), color)
        return fcn.utils.get_tile_image([viz_true, viz_pred], (1, 2))

    trainer.extend(
        fcn.training.extensions.ImageVisualizer(
            chainer.iterators.SerialIterator(
                dataset_train, batch_size=1, shuffle=False),
            model, visualize_ss, device=gpus[0]),
        trigger=(10, 'iteration'),
        invoke_before_training=True,
    )

    model_name = model.__class__.__name__
    trainer.extend(extensions.dump_graph(
        'main/loss', out_name='%s.dot' % model_name))
    trainer.extend(extensions.snapshot(
        savefun=chainer.serializers.hdf5.save_hdf5,
        filename='%s_trainer_iter_{.updater.iteration}.h5' % model_name,
        trigger=(interval_eval, 'iteration')))
    trainer.extend(extensions.snapshot_object(
        model,
        savefun=chainer.serializers.hdf5.save_hdf5,
        filename='%s_model_iter_{.updater.iteration}.h5' % model_name,
        trigger=(interval_eval, 'iteration')))
    trainer.extend(extensions.LogReport(
        trigger=(interval_log, 'iteration'), log_name='log.json'))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/loss', 'validation/main/loss',
        'main/loss_cls', 'validation/main/loss_cls',
        'main/loss_seg', 'validation/main/loss_seg',
        'main/accuracy', 'validation/main/accuracy',
        'main/cls_iu', 'validation/main/cls_iu',
        'main/ins_iu', 'validation/main/ins_iu',
        'elapsed_time',
    ]))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if resume:
        if resume.endswith('npz'):
            chainer.serializers.load_npz(resume, trainer)
        else:
            chainer.serializers.load_hdf5(resume, trainer)

    return trainer


this_dir = osp.dirname(osp.realpath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--out',
        help='default: logs/<timestamp> or parent dir of `resume`')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='default: 0')
    parser.add_argument('--resume')
    args = parser.parse_args()

    gpu = args.gpu
    out = args.out
    resume = args.resume

    dataset_train = rfcn.datasets.PascalInstanceSegmentationDataset(
        'train', rp=True)
    dataset_val = rfcn.datasets.PascalInstanceSegmentationDataset(
        'val', rp=True)

    optimizer = chainer.optimizers.Adam(alpha=0.002)

    trainer = get_trainer(
        dataset_train,
        dataset_val,
        optimizer=optimizer,
        gpu=gpu,
        max_iter=10000,
        out=out,
        resume=resume,
        interval_log=10,
        interval_eval=1000,
    )
    trainer.run()


if __name__ == '__main__':
    main()
