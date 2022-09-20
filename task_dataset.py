# author = liuwei
# date = 2022-09-19

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import nltk

class EmbedDataset(Dataset):
    """
    Args:
        file_name,
        params:
    """
    def __init__(self, file_name, params):
        self.file_name = file_name
        self.max_seq_length = params["max_seq_length"]
        self.labels = params["labels"]
        self.vocab = params["vocab"]
        self.text_key = params["text_key"]
        self.label_key = params["label_key"]


        self.init_dataset()

    def convert_tokens_to_ids(self, tokens):
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab.index(token))
            else:
                tokens.append(self.vocab.index("<unk>")) # 0 as <pad> id

        return token_ids

    def init_dataset(self):
        all_input_ids = []
        all_seq_lengths = []
        all_label_ids = []
        with open(self.file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)

                    text = sample[self.text_key]
                    label = sample[self.label_key]

                    tokens = nltk.word_tokenize(text)
                    if len(tokens) > self.max_seq_length:
                        tokens = tokens[:self.max_seq_length]
                    token_ids = self.convert_tokens_to_ids(tokens)
                    seq_length = len(token_ids)
                    label_id = self.labels.index(label)

                    input_ids = np.zeros(self.max_seq_length, dtype=np.int)
                    input_ids[:len(token_ids)] = token_ids

                    all_input_ids.append(input_ids)
                    all_seq_lengths.append(seq_length)
                    all_label_ids.append(label_id)
        assert len(all_input_ids) == len(all_seq_lengths), (len(all_input_ids), len(all_seq_lengths))
        assert len(all_input_ids) == len(all_label_ids), (len(all_input_ids), len(all_label_ids))

        all_input_ids = np.array(all_input_ids)
        all_seq_lengths = np.array(all_seq_lengths)
        all_label_ids = np.array(all_label_ids)

        self.input_ids = all_input_ids
        self.seq_lengths = all_seq_lengths
        self.label_ids = all_label_ids
        self.total_size = self.input_ids.shape[0]

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.seq_lengths[index]),
            torch.tensor(self.label_ids[index])
        )



