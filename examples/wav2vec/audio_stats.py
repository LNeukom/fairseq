import argparse
import glob
import os

import numpy as np

from mutagen.mp3 import MP3
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', metavar='DIR', help='root directory containing flac files to index')
    parser.add_argument('--ext', default='mp3', type=str, metavar='EXT', help='extension to look for')
    return parser


def main(args):
    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, '**/*.' + args.ext)

    lengths = []
    for fname in tqdm(glob.glob(search_path, recursive=True)):
        file_path = os.path.realpath(fname)
        try:
            audio = MP3(file_path)
            lengths.append(audio.info.length)
        except Exception as e:
            pass
    print("Audio stats:")
    print("Mean length: ", np.mean(lengths), "Std:", np.std(lengths))
    print("Median length: ", np.median(lengths))
    print("Min length: ", np.min(lengths))
    print("Max length: ", np.max(lengths))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)