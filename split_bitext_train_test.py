import os

import numpy as np


def write_split(name, lines):
    print(f"Write {name} with {len(lines)} lines.")
    with open(name, 'w') as f:
        f.writelines(lines)
    os.system(f"gzip {name}")


def run():
    test_size = 2000
    train_sizes = [10_000, 100_000, 1_000_000]
    name = "jrc"
    files = ["jrc.de", "jrc.fr"]
    langs = ["de", "fr"]
    skip_first_line = True

    corpora = []
    for file in files:
        with open(file, 'r') as f:
            corpora.append(f.readlines())
    assert all([len(c) == len(corpora[0]) for c in corpora])
    print(f"Loaded {len(corpora)} text corpora with {len(corpora[0])} lines each.")

    if skip_first_line:
        corpora = [c[1:] for c in corpora]

    indices = np.arange(len(corpora[0]))
    np.random.shuffle(indices)

    for lang, corpus in zip(langs, corpora):
        write_split(f'{name}.test.{lang}', [corpus[i] for i in indices[:test_size]])
        for train_size in train_sizes:
            if train_size >= 1_000_000:
                tag = f"{int(train_size//1_000_000)}m"
            elif train_size >= 1_000:
                tag = f"{int(train_size//1_000)}k"
            else:
                tag = f"{train_size}"
            write_split(f'{name}.train{tag}.{lang}', [corpus[i] for i in indices[test_size:(test_size+train_size)]])
            # corpus[indices[test_size:(test_size+train_size)]]


if __name__ == '__main__':
    run()
