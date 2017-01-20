import argparse
import os
import os.path as osp

import chainer
from chainer import cuda
import fcn
from fcn.models import FCN8s
import numpy as np
import skimage.io

import rfcn
from rfcn.datasets import PascalInstanceSegmentationDataset
from rfcn.external.faster_rcnn.models.faster_rcnn import FasterRCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', default='logs/pascal_val_inference')
    args = parser.parse_args()

    gpu = args.gpu
    out_dir = args.out

    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    if gpu >= 0:
        cuda.get_device(gpu).use()

    dataset = PascalInstanceSegmentationDataset('val')
    class_names = dataset.class_names
    n_class = len(dataset.class_names)

    rcnn_path = rfcn.data.download_faster_rcnn_chainermodel()
    rcnn_model = FasterRCNN(gpu=gpu, num_classes=n_class)
    chainer.serializers.load_npz(rcnn_path, rcnn_model)
    rcnn_model.train = False
    if gpu >= 0:
        rcnn_model.to_gpu()

    fcn_path = fcn.data.download_fcn8s_from_caffe_chainermodel()
    fcn_model = FCN8s(n_class=n_class)
    chainer.serializers.load_hdf5(fcn_path, fcn_model)
    fcn_model.train = False
    if gpu >= 0:
        fcn_model.to_gpu()

    csv_file = osp.join(out_dir, 'results.csv')
    csv_template = '{index},{iu_cls},{iu_ins},{iu_mean}\n'
    with open(csv_file, 'w') as f:
        f.write(csv_template.replace('{', '').replace('}', ''))
    for dataset_index in xrange(len(dataset)):
        out_sub_dir = osp.join(out_dir, str(dataset_index))
        if not osp.exists(out_sub_dir):
            os.makedirs(out_sub_dir)

        datum, lbl_ins_true, lbl_cls_true = dataset.get_example(dataset_index)
        img = dataset.datum_to_img(datum)

        x_data = np.expand_dims(datum, axis=0)
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data, device=gpu)
        x = chainer.Variable(x_data, volatile=True)

        cls_score, bbox_pred = rcnn_model(
            x, np.array([[x.shape[2], x.shape[3], 1.]]))
        cls_score = cuda.to_cpu(cls_score.data)
        bbox_pred = cuda.to_cpu(bbox_pred)

        fcn_model(x)
        score = cuda.to_cpu(fcn_model.score.data)
        lbl_cls = np.argmax(score[0], axis=0)

        rois = []
        roi_clss = []
        for cls_id, cls_name in enumerate(dataset.class_names):
            _cls = cls_score[:, cls_id][:, np.newaxis]
            _bbx = bbox_pred[:, cls_id*4:(cls_id+1)*4]
            dets = np.hstack((_bbx, _cls))
            keep = rfcn.utils.nms(_bbx, thresh=0.3, scores=_cls[:, 0])
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= 0.8)[0]
            rois.extend(dets[inds, :4].tolist())
            roi_clss.extend([cls_id] * len(inds))
        rois = np.array(rois, dtype=np.int64)
        roi_clss = np.array(roi_clss, dtype=np.int32)

        roi_sizes = (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])
        keep = np.bitwise_and(roi_sizes > 0, roi_clss != 0)
        rois = rois[keep]
        roi_clss = roi_clss[keep]
        roi_sizes = roi_sizes[keep]

        label_titles = dict(zip(np.arange(n_class), dataset.class_names))
        viz = fcn.utils.draw_label(lbl_cls, img, n_class=n_class,
                                   label_titles=label_titles)
        skimage.io.imsave(
            osp.join(out_sub_dir, 'object_segmentation.jpg'), viz)

        viz = rfcn.utils.draw_instance_boxes(img, rois, roi_clss, n_class,
                                             captions=class_names[roi_clss])
        skimage.io.imsave(osp.join(out_sub_dir, 'object_detection.jpg'), viz)

        lbl_ins = np.zeros_like(lbl_cls)
        lbl_ins[...] = -1

        for i in np.argsort(roi_sizes):
            x1, y1, x2, y2 = map(int, rois[i])
            roi_cls = roi_clss[i]

            roi_lbl_ins = lbl_ins[y1:y2, x1:x2]
            roi_lbl_cls = lbl_cls[y1:y2, x1:x2]
            mask_ins = roi_lbl_cls == roi_cls
            mask_ins = np.bitwise_and(mask_ins, roi_lbl_ins == -1)

            lbl_ins[y1:y2, x1:x2][mask_ins] = i

        viz = rfcn.utils.visualize_instance_segmentation(
            lbl_ins, lbl_cls, img, dataset.class_names)
        skimage.io.imsave(
            osp.join(out_sub_dir, 'instance_segmentation.jpg'), viz)

        iu_cls = fcn.utils.label_accuracy_score(
            lbl_cls_true, lbl_cls, n_class)[2]
        iu_ins = rfcn.utils.instance_label_accuracy_score(
            lbl_ins_true, lbl_ins)
        iu_mean = (iu_cls + iu_ins) / 2.0
        with open(csv_file, 'a') as f:
            f.write(csv_template.format(
                index=dataset_index,
                iu_cls=iu_cls,
                iu_ins=iu_ins,
                iu_mean=iu_mean,
            ))


if __name__ == '__main__':
    main()
