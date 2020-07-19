#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import soundfile
import random


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', metavar='DIR', help='root directory containing flac files to index')
    parser.add_argument('--valid-percent', default=0.01, type=float, metavar='D',
                        help='percentage of data to use as validation set (between 0 and 1)')
    parser.add_argument('--dest', default='.', type=str, metavar='DIR', help='output directory')
    parser.add_argument('--ext', default='flac', type=str, metavar='EXT', help='extension to look for')
    parser.add_argument('--min-duration', default=10., type=float, metavar='D', help='min audio duration')
    parser.add_argument('--max-duration', default=40., type=float, metavar='D', help='max audio duration')
    parser.add_argument('--seed', default=42, type=int, metavar='N', help='random seed')
    parser.add_argument('--path-must-contain', default=None, type=str, metavar='FRAG',
                        help='if set, path must contain this substring for a file to be included in the manifest')
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, '**/*.' + args.ext)
    rand = random.Random(args.seed)

    with open(os.path.join(args.dest, 'train.tsv'), 'w') as train_f, open(
            os.path.join(args.dest, 'valid.tsv'), 'w') as valid_f:
        print(dir_path, file=train_f)
        print(dir_path, file=valid_f)

        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)

            if args.path_must_contain and args.path_must_contain not in file_path:
                print(f"Ignore {fname}: path does not contain {args.path_must_contain}")
                continue

            file_info = soundfile.info(fname)

            if file_info.duration < args.min_duration:
                print(f"Ignore {fname}: duration {file_info.duration:.2f} < min_duration {args.min_duration:.2f}")
                continue

            if file_info.duration > args.max_duration:
                print(f"Ignore {fname}: duration {file_info.duration:.2f} > max_duration {args.max_duration:.2f}")
                continue

            dest = train_f if rand.random() > args.valid_percent else valid_f
            print('{}\t{}'.format(os.path.relpath(file_path, dir_path), file_info.frames), file=dest)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
