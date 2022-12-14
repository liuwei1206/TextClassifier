# author = liuwei
# date = 2022-09-20

import os
import json

import numpy as np
import torch
import torch.nn as nn

from modules import Embedding, BiLSTM

class TextLSTM(nn.Module):
    def __init__(self, params):
        super(TextLSTM, self).__init__()

        vocab = params["vocab"]
        embed_file = params["embed_file"]
        data_dir = params["data_dir"]
        self.input_dim = params["input_dim"]
        self.hidden_size = params["hidden_size"]
        self.num_layers = params["num_layers"]
        self.num_labels = params["num_labels"]
        self.dropout = nn.Dropout(p=params["dropout"])
        self.word_embedding = Embedding(vocab, self.input_dim, embed_file, data_dir)
        self.bilstm = BiLSTM(self.input_dim, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.classifier = nn.Linear(self.hidden_size // 4, self.num_labels)

        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(mean=0.0, std=0.02)
        self.fc.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

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
        f_last_state = f_lstm_out[:, 0, :].squeeze(dim=1)
        b_last_state = b_lstm_out[:, 0, :].squeeze(dim=1)
        lstm_out = torch.cat((f_last_state, b_last_state), dim=-1) # [B, 2D]
        lstm_out = self.dropout(lstm_out)
        fc_out = self.fc(lstm_out)
        fc_out = self.dropout(fc_out)
        logits = self.classifier(fc_out)

        preds = torch.argmax(logits, dim=-1) # [N]
        outputs = (preds, )
        if flag.lower() == "train":
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs
