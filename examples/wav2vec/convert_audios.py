import argparse
import glob
import os
import subprocess

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', metavar='DIR', help='root directory containing flac files to index')
    parser.add_argument('--src-ext', default='mp3', type=str, metavar='EXT', help='extension to look for')
    parser.add_argument('--dst-ext', default='flac', type=str, metavar='EXT', help='extension to convert to')
    return parser


def main(args):
    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, '**/*.' + args.src_ext)

    for src_fname in tqdm(glob.glob(search_path, recursive=True)):
        src_file_path = os.path.realpath(src_fname)
        dst_fname = src_fname.replace(args.src_ext, args.dst_ext)
        dst_file_path = os.path.realpath(dst_fname)

        if not os.path.exists(dst_file_path):
            try:
                process = subprocess.Popen(
                    f"ffmpeg -hide_banner -loglevel panic -y -i {src_file_path} {dst_file_path}",
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, error = process.communicate()
                process.wait()
                if process.returncode != 0:
                    print(f"Failed to convert {src_file_path} to {dst_file_path}: {process.returncode:d} {output} "
                          f"{error}")
            except Exception as e:
                print(f"Failed to convert {src_file_path} to {dst_file_path}: {e}")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
