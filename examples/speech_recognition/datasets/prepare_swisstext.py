import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import sentencepiece as spm

from examples.speech_recognition.datasets import asr_prep_json
from fairseq.data import Dictionary


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-path', metavar='DIR', help='root directory containing raw dataset files')
    parser.add_argument('--wav2vec-path', metavar='DIR', default=None, help='root directory containing wav2vec encoded audio files')
    parser.add_argument('--out-path', metavar='DIR', help='output directory')
    parser.add_argument('--dev-split-prop', default=0.1, type=float, help='dev set split proportion')
    parser.add_argument('--train-dir', default='train_70', type=str, help='just for consitency with librespeech')
    return parser


def load_cv_set(path, split):
    return pd.read_csv(os.path.join(path, f"{split}.tsv"), sep='\t')


def split_set(data_set, proportion: float):
    split_mask = np.random.rand(len(data_set)) < proportion
    return data_set[~split_mask], data_set[split_mask]


def train_bpe(data_sets: list, out_path: str, train_dir: str, nbpe=5000, bpemode='unigram'):
    bpe_base_path = os.path.join(out_path, 'data', 'lang_char')

    encoded = os.path.join(bpe_base_path, f'{train_dir}_{bpemode}{nbpe}_encoded.txt')
    bpemodel = os.path.join(bpe_base_path, f'{train_dir}_{bpemode}{nbpe}')
    bpe_dict = os.path.join(bpe_base_path, f'{train_dir}_{bpemode}{nbpe}_units.txt')
    fairseq_dict = os.path.join(bpe_base_path, f'{train_dir}_{bpemode}{nbpe}_fairseq_dict.txt')

    print(f'dictionary: {bpe_dict}')
    print(f'Dictionary preparation')
    os.makedirs(bpe_base_path, exist_ok=True)

    # prepare bpe train set
    input_text = os.path.join(bpe_base_path, 'input.txt')
    with open(input_text, 'w') as f:
        for data_set in data_sets:
            f.writelines(data_set['sentence'].tolist())

    # train bpe
    subprocess.run([
        'spm_train', f'--input={input_text}', f'--vocab_size={nbpe}', f'--model_type={bpemode}',
        f'--model_prefix={bpemodel}', '--input_sentence_size=100000000', '--unk_id=3', '--eos_id=2',
        '--pad_id=1', '--bos_id=-1', '--character_coverage=1'
    ])

    # encode bpe
    with open(input_text, 'r') as f_in:
        with open(encoded, 'w') as f_out:
            subprocess.run([
                'spm_encode', f'--model={bpemodel}.model', '--output_format=piece'
            ], stdin=f_in, stdout=f_out)

    encoded_tokens = []
    with open(encoded, 'r') as f:
        for line in f:
            encoded_tokens.extend(line.split())
    encoded_tokens = sorted(encoded_tokens)
    unqiue_tokens, counts = np.unique(encoded_tokens, unique_counts=True)

    with open(bpe_dict, 'w') as f:
        f.writelines(['<unk> 3', '</s> 2', '<pad> 1'])
        for i, token in enumerate(unqiue_tokens):
            f.write(f"{token} {i+4}")

    with open(fairseq_dict, 'w') as f:
        for token, token_count in zip(unqiue_tokens, counts):
            f.write(f"{token} {token_count}")

    shutil.copyfile(fairseq_dict, os.path.join(out_path, 'dict.txt'))
    shutil.copyfile(f'{bpemodel}.model', os.path.join(out_path, 'spm.model'))

    sp = spm.SentencePieceProcessor()
    sp.Load(f'{bpemodel}.model')

    tgt_dict = Dictionary.load(fairseq_dict)

    return sp, tgt_dict


def process_sample(raw_path, wav2vec_path, clip, sentence, speaker_id, sp, tgt_dict):
    import h5py
    audio_path = os.path.join(raw_path, 'clips', clip)
    utt_id = f'{speaker_id}-{os.path.splitext(audio_path)[0]}-0000'  # fake librispeech id: <speaker>-<chapter>-<seq>
    if wav2vec_path is not None:
        wav2vec_path = os.path.join(wav2vec_path, 'clips', f'{utt_id}.h5context')
    sample = asr_prep_json.process_sample(audio_path, sentence, utt_id, sp, tgt_dict)
    if wav2vec_path is not None:
        with h5py.File(wav2vec_path, 'r') as f:
            sample[utt_id]['input']['wav2vec_num_tokens'] = f['info'][1]
        sample[utt_id]['input']['wav2vec_path'] = wav2vec_path
    return sample


def prepare_json(split: pd.DataFrame, raw_path: str, wav2vec_path: Optional[str], output: str, sp, tgt_dict):
    utts = {}
    num_cpu = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpu) as executor:
        futures = []
        for index, row in split.iterrows():
            futures.append(executor.submit(process_sample, raw_path, wav2vec_path, row['sentence'], row['client_id'], sp, tgt_dict))
        for future in concurrent.futures.as_completed(futures):
            try:
                data = future.result()
            except Exception as exc:
                print('generated an exception: ', exc)
            else:
                utts.update(data)
    with open(output, 'w') as f:
        json.dump({"utts": utts}, f, indent=4)


def main(args):
    os.makedirs(args.out_path, exist_ok=True)

    train_set = load_cv_set(args.raw_path, 'train')
    train_set, dev_set = split_set(train_set, args.dev_split_prop)
    test_private_set = load_cv_set(args.raw_path, 'test_private')
    test_public_set = load_cv_set(args.raw_path, 'test_public')

    sp, tgt_dict = train_bpe([train_set], args.out_path, args.train_dir, nbpe=5000, bpemode='unigram')

    prepare_json(train_set, args.raw_path, args.wav2vec_path,
                 os.path.join(args.out_path, 'train.json'), sp, tgt_dict)
    prepare_json(dev_set, args.raw_path, args.wav2vec_path,
                 os.path.join(args.out_path, 'dev.json'), sp, tgt_dict)
    prepare_json(test_private_set, args.raw_path, args.wav2vec_path,
                 os.path.join(args.out_path, 'test_private.json'), sp, tgt_dict)
    prepare_json(test_public_set, args.raw_path, args.wav2vec_path,
                 os.path.join(args.out_path, 'test_public.json'), sp, tgt_dict)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
