#!/usr/bin/env python

from __future__ import print_function

import argparse
import datetime
import os
import os.path as osp
import sys

import chainer
from chainer.training import extensions
import fcn
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
    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=1)
    iter_val = chainer.iterators.SerialIterator(
        dataset_val, batch_size=1, repeat=False, shuffle=False)

    # 2. model
    vgg_path = fcn.data.download_vgg16_chainermodel()
    vgg = fcn.models.VGG16()
    chainer.serializers.load_hdf5(vgg_path, vgg)

    n_classes = len(dataset_train.class_names) - 1
    model = rfcn.models.FCISVGG_SS(C=n_classes, k=7)
    model.train = True
    fcn.utils.copy_chainermodel(vgg, model.trunk)

    chainer.cuda.get_device(gpu).use()
    model.to_gpu()

    # 3. optimizer
    optimizer.setup(model)

    # 4. trainer
    updater = chainer.training.StandardUpdater(
        iter_train, optimizer, device=gpu)
    trainer = chainer.training.Trainer(
        updater, (max_iter, 'iteration'), out=out)

    trainer.extend(
        fcn.training.extensions.TestModeEvaluator(
            iter_val, model, device=gpu),
        trigger=(interval_eval, 'iteration'),
        invoke_before_training=False,
    )

    def visualize_ss(target):
        datum = chainer.cuda.to_cpu(target.x.data[0])
        img = dataset_val.datum_to_img(datum).copy()
        class_names = dataset_val.class_names
        n_class = len(class_names)
        rois = target.rois
        # visualize true
        lbl_ins = target.lbl_ins
        lbl_cls = target.lbl_cls
        roi_clss = target.roi_clss
        viz_lbl_true = rfcn.utils.visualize_instance_segmentation(
            lbl_ins, lbl_cls, img, class_names)
        viz_rois_all_true = rfcn.utils.draw_instance_boxes(
            img, rois, roi_clss, n_class, bg_class=-1)
        viz_rois_pos_true = rfcn.utils.draw_instance_boxes(
            img, rois, roi_clss, n_class, bg_class=0)
        viz_true = fcn.utils.get_tile_image(
            [viz_lbl_true, viz_rois_all_true, viz_rois_pos_true], (1, 3))
        # visualize prediction
        lbl_ins_pred = target.lbl_ins_pred
        lbl_cls_pred = target.lbl_cls_pred
        roi_clss_pred = target.roi_clss_pred
        viz_lbl_pred = rfcn.utils.visualize_instance_segmentation(
            lbl_ins_pred, lbl_cls_pred, img, class_names)
        viz_rois_all_pred = rfcn.utils.draw_instance_boxes(
            img, rois, roi_clss_pred, n_class, bg_class=-1)
        viz_rois_pos_pred = rfcn.utils.draw_instance_boxes(
            img, rois, roi_clss_pred, n_class, bg_class=0)
        viz_pred = fcn.utils.get_tile_image(
            [viz_lbl_pred, viz_rois_all_pred, viz_rois_pos_pred], (1, 3))
        return fcn.utils.get_tile_image([viz_true, viz_pred], (2, 1))

    os.mkdir(osp.join(out, 'viz_train'))
    trainer.extend(
        fcn.training.extensions.ImageVisualizer(
            chainer.iterators.SerialIterator(
                dataset_train, batch_size=1, shuffle=False),
            model,
            visualize_ss,
            out='viz_train/{.updater.iteration}.jpg',
            device=gpu,
        ),
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

    dataset_train = rfcn.datasets.PascalInstanceSegmentationRPDataset(
        'train', negative_ratio=0.1)
    dataset_val = rfcn.datasets.PascalInstanceSegmentationRPDataset(
        'val', negative_ratio=0.1)

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
