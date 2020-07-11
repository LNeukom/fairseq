import argparse
import glob
import multiprocessing
import os
import subprocess

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', metavar='DIR', help='root directory containing flac files to index')
    parser.add_argument('--src-ext', default='mp3', type=str, metavar='EXT', help='extension to look for')
    parser.add_argument('--dst-ext', default='flac', type=str, metavar='EXT', help='extension to convert to')
    return parser


def convert_audio(sr_path, dst_path):
    if not os.path.exists(dst_path):
        try:
            process = subprocess.Popen(f"ffmpeg -hide_banner -loglevel panic -y -i {sr_path} {dst_path}",
                                       shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            process.wait()
            if process.returncode != 0:
                print(f"Failed to convert {sr_path} to {dst_path}: {process.returncode:d} {output} {error}")
        except Exception as e:
            print(f"Failed to convert {sr_path} to {dst_path}: {e}")
    return dst_path


def main(args):
    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, '**/*.' + args.src_ext)
    src_paths = glob.glob(search_path, recursive=True)

    progress_bar = tqdm(total=100)

    def update_progress_bar(*_):
        progress_bar.update()

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        for src_path in src_paths:
            pool.apply_async(convert_audio, args=(src_path,), callback=update_progress_bar)
        pool.join()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
