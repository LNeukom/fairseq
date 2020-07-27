import os

import numpy as np

from fairseq.data import FairseqDataset
from .collaters import Seq2SeqCollater


class W2VAsrDataset(FairseqDataset):
    """
    A dataset representing speech and corresponding transcription.

    Args:
        emb_paths: (List[str]): A list of str with paths to audio embedding files.
        emb_num_tokens (List[int]): A list of int containing the number of audio embedding tokens per file.
        tgt (List[torch.LongTensor]): A list of LongTensors containing the indices
            of target transcriptions.
        tgt_dict (~fairseq.data.Dictionary): target vocabulary.
        ids (List[str]): A list of utterance IDs.
        speakers (List[str]): A list of speakers corresponding to utterances.
    """

    def __init__(
        self, emb_paths, emb_num_tokens, tgt, tgt_dict, ids, speakers
    ):
        assert len(emb_paths) > 0
        assert len(emb_paths) == len(emb_num_tokens)
        assert len(emb_paths) == len(tgt)
        assert len(emb_paths) == len(ids)
        assert len(emb_paths) == len(speakers)
        self.emb_paths = emb_paths
        self.emb_num_tokens = emb_num_tokens
        self.tgt_dict = tgt_dict
        self.tgt = tgt
        self.ids = ids
        self.speakers = speakers
        self.s2s_collater = Seq2SeqCollater(
            0, 1, pad_index=self.tgt_dict.pad(),
            eos_index=self.tgt_dict.eos(), move_eos_to_beginning=True
        )

    def __getitem__(self, index):
        import h5py
        tgt_item = self.tgt[index] if self.tgt is not None else None
        path = self.emb_paths[index]
        if not os.path.exists(path):
            raise FileNotFoundError("Audio file not found: {}".format(path))
        with h5py.File(self.fname, "r") as f:
            features = f["features"]
        return {"id": index, "data": [features, tgt_item]}

    def __len__(self):
        return len(self.emb_paths)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return self.s2s_collater.collate(samples)

    def num_tokens(self, index):
        return self.emb_num_tokens[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.self.emb_num_tokens[index],
            len(self.tgt[index]) if self.tgt is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))