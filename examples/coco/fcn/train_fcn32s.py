#!/usr/bin/env python

import argparse

import chainer
import fcn

import rfcn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-o', '--out', default='logs/latest')
    parser.add_argument('--resume')
    args = parser.parse_args()

    gpu = args.gpu
    out = args.out
    resume = args.resume
    max_iter = 100000

    trainer = fcn.trainers.fcn32s.get_trainer(
        dataset_class=rfcn.datasets.CocoSegmentaionDataset,
        gpu=gpu,
        max_iter=max_iter,
        out=out,
        resume=resume,
        interval_log=10,
        interval_eval=1000,
        optimizer=chainer.optimizers.Adam(),
    )
    trainer.run()


if __name__ == '__main__':
    main()
