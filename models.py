# author = liuwei
# date = 2022-09-20

import os
import json

import numpy as np
import torch
import torch.nn as nn

from modules import Embedding, BiLSMT

class TextLSTM(nn.Module):
    def __init__(self, params):
        super(TextLSTM, self).__init__()

        vocab = params["vocab"]
        embed_file = params["embed_file"]
        self.input_dim = params["input_dim"]
        self.hidden_size = params["hidden_size"]
        self.num_layers = params["num_layers"]
        self.num_labels = params["num_labels"]
        self.dropout = nn.Dropout(p=params["dropout"])
        self.word_embedding = Embedding(vocab, self.input_dim, embed_file)
        self.bilstm = BiLSMT(self.input_dim, self.hidden_size, self.num_layers)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids,
        seq_lengths,
        labels,
        flag="Train"
    ):
        """
        Args:
            input_ids: [batch_size, seq_length]
            seq_lengths: [batch_size]
            labels: [batch_size]
            flag: Train or Eval
        """
        input_embeddings = self.word_embedding(input_ids)
        input_embeddings = self.dropout(input_embeddings)
        _, f_lstm_out, b_lstm_out = self.bilstm(input_embeddings, seq_lengths)
        last_index = seq_lengths.view(-1, 1, 1)
        last_index = last_index.repeat(1, 1, self.hidden_size // 2)
        f_last_state = torch.gather(f_lstm_out, dim=1, index=last_index).squeeze(dim=1) # [B, D]
        b_last_state = b_lstm_out[:, 0, :].squeeze(dim=1)
        lstm_out = torch.cat((f_last_state, b_last_state), dim=-1) # [B, 2D]
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)

        preds = torch.argmax(logits, dim=-1) # [N]
        outputs = (preds, )
        if flag.lower() == "train":
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs
