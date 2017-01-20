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
    class_names = dataset.class_names
    n_class = len(class_names)
    rois = model.rois
    # visualize true
    lbl_ins = model.lbl_ins
    lbl_cls = model.lbl_cls
    #roi_clss = model.roi_clss
    viz_lbl_true = rfcn.utils.visualize_instance_segmentation(
        lbl_ins, lbl_cls, img, class_names)
    #viz_rois_all_true = rfcn.utils.draw_instance_boxes(
    #    img, rois, roi_clss, n_class, bg_class=-1)
    #viz_rois_pos_true = rfcn.utils.draw_instance_boxes(
    #    img, rois, roi_clss, n_class, bg_class=0)
    #viz_true = fcn.utils.get_tile_image(
    #    [viz_lbl_true, viz_rois_all_true, viz_rois_pos_true], (1, 3))
    # visualize prediction
    lbl_ins_pred = model.lbl_ins_pred
    lbl_cls_pred = model.lbl_cls_pred
    #roi_clss_pred = model.roi_clss_pred
    viz_lbl_pred = rfcn.utils.visualize_instance_segmentation(
        lbl_ins_pred, lbl_cls_pred, img, class_names)
    #viz_rois_all_pred = rfcn.utils.draw_instance_boxes(
    #    img, rois, roi_clss_pred, n_class, bg_class=-1)
    #viz_rois_pos_pred = rfcn.utils.draw_instance_boxes(
    #    img, rois, roi_clss_pred, n_class, bg_class=0)
    #viz_pred = fcn.utils.get_tile_image(
    #    [viz_lbl_pred, viz_rois_all_pred, viz_rois_pos_pred], (1, 3))
    #return fcn.utils.get_tile_image([viz_true, viz_pred], (2, 1))
    return fcn.utils.get_tile_image([viz_lbl_true, viz_lbl_pred], (2, 1))


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
