# author = liuwei
# date = 2022-09-20

import os
import json
from utils import reverse_padded_sequence, build_embedding_of_corpus

import numpy as np
import torch
import torch.nn as nn

class Embedding(torch.nn.Module):
    def __init__(self, vocab, embed_dim, embed_file, data_dir):
        super(Embedding, self).__init__()

        self.vocab = vocab
        self.embed_dim = embed_dim
        self.embed_file = embed_file
        self.data_dir = data_dir
        self.embedding = nn.Embedding(len(vocab), embed_dim)

        self.init_embedding()

    def init_embedding(self):
        corpus_embed = build_embedding_of_corpus(self.embed_file, self.vocab, self.embed_dim, self.data_dir)
        assert corpus_embed.shape[0] == len(self.vocab), (corpus_embed.shape[0], len(self.vocab))
        assert corpus_embed.shape[1] == self.embed_dim, (corpus_embed.shape[1], self.embed_dim)

        self.embedding.weight.data.copy_(torch.from_numpy(corpus_embed))

    def forward(self, input_ids):
        return self.embedding(input_ids)


class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTM, self).__init__()

        self.f_lstm = nn.LSTM(input_size, hidden_size // 2, num_layers=num_layers, batch_first=True)
        self.b_lstm = nn.LSTM(input_size, hidden_size // 2, num_layers=num_layers, batch_first=True)
        # self.f_lstm.reset_parameters()
        # self.b_lstm.reset_parameters()

    def forward(self, inputs, seq_lengths):
        """
        Args:
            inputs: [batch_size, seq_length, input_size]
            seq_lengths: [batch_size]
        """
        lengths = list(map(int, seq_lengths))
        reversed_inputs = reverse_padded_sequence(inputs, lengths)

        f_lstm_out, f_hidden = self.f_lstm(inputs)
        b_lstm_out, b_hidden = self.b_lstm(reversed_inputs)
        b_lstm_out = reverse_padded_sequence(b_lstm_out, lengths)
        lstm_out = torch.cat((f_lstm_out, b_lstm_out), dim=-1)
        f_lstm_out = reverse_padded_sequence(f_lstm_out, lengths)

        return lstm_out, f_lstm_out, b_lstm_out
