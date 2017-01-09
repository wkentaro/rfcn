#!/usr/bin/env python

from __future__ import division

import argparse
import collections
import json

import os.path as osp

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import pandas as pd


def learning_curve(json_file):
    if not osp.exists(json_file):
        print('JSON file does not exist: {}'.format(json_file))
        quit(1)
    df = pd.DataFrame(json.load(open(json_file, 'r')))

    plt.style.use('seaborn-paper')

    plt.figure(figsize=(12, 6), dpi=500)

    # initialize DataFrame for train
    columns = [
        'iteration',
        'main/loss',
        'main/iu',
    ]
    df_train = df[columns]
    # get min/max
    row_max = df.max()
    row_min = df.min()
    # make smooth the learning curve with iteration step
    iteration_step = 10
    df_train_stat = []
    stat = collections.defaultdict(list)
    for index, row in df_train.iterrows():
        for col in row.keys():
            value = row[col]
            stat[col].append(value)
        if int(row['iteration']) % iteration_step == 0:
            means = [sum(stat[col]) / len(stat[col]) for col in row.keys()]
            means[0] = row['iteration']  # iteration_step is the representative
            df_train_stat.append(means)
            stat = collections.defaultdict(list)
    df_train = pd.DataFrame(df_train_stat, columns=df_train.columns)

    # initialize DataFrame for val
    columns = [
        'iteration',
        'validation/main/loss',
        'validation/main/iu',
    ]
    try:
        df_val = df[columns]
        df_val = df_val.dropna()
    except KeyError:
        df_val = None

    #########
    # TRAIN #
    #########

    # train loss
    plt.subplot(221)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.semilogy(df_train['iteration'], df_train['main/loss'], '-',
                 markersize=1, color='r', alpha=.5, label='train loss')
    plt.grid(True)
    plt.xlim((row_min['iteration'], row_max['iteration']))
    plt.ylim((min(row_min['main/loss'], row_min['validation/main/loss']),
              max(row_max['main/loss'], row_max['validation/main/loss'])))
    plt.xlabel('iteration')
    plt.ylabel('train loss')

    # train iu
    plt.subplot(222)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.plot(df_train['iteration'], df_train['main/iu'],
             '-', markersize=1, color='b', alpha=.5,
             label='train iu')
    plt.grid(True)
    plt.xlim((row_min['iteration'], row_max['iteration']))
    plt.ylim((0, 1.0))
    plt.xlabel('iteration')
    plt.ylabel('train iu')

    if df_val is not None:
        #######
        # VAL #
        #######

        # val loss
        plt.subplot(223)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.semilogy(df_val['iteration'], df_val['validation/main/loss'],
                     'o-', color='r', alpha=.5, label='val loss')
        plt.grid(True)
        plt.xlim((row_min['iteration'], row_max['iteration']))
        plt.ylim((min(row_min['main/loss'], row_min['validation/main/loss']),
                  max(row_max['main/loss'], row_max['validation/main/loss'])))
        plt.xlabel('iteration')
        plt.ylabel('val loss')

        # val iu
        plt.subplot(224)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.plot(df_val['iteration'],
                 df_val['validation/main/iu'],
                 'o-', color='b', alpha=.5, label='val iu')
        plt.grid(True)
        plt.xlim((row_min['iteration'], row_max['iteration']))
        plt.ylim((0, 1.0))
        plt.xlabel('iteration')
        plt.ylabel('val iu')

    fig_file = '{}.png'.format(osp.splitext(json_file)[0])
    plt.savefig(fig_file)
    print('Saved to %s' % fig_file)

    plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()

    json_file = args.json_file

    learning_curve(json_file)


if __name__ == '__main__':
    main()
