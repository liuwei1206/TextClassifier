# author = liuwei
# date = 2022-09-19

import os
import json
import pickle
import random

import nltk
import numpy as np
import torch

random.seed(106524)
np.random.seed(106524)

def get_vocab_from_corpus(data_dir, item_key="text", require_unk=False):
    """
    Args:
        data_dir: the data dir, contain train.json, dev.json, test.json
        item_key: the key in sample's dict
        require_unk: require a unk token or not
    Returns:
        vocab: a list of words
    """
    vocab = set()
    for file_name in ["train.json", "dev.json", "test.json"]:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    text = sample[item_key]
                    if item_key == "relations":
                        tokens = text
                    else:
                        tokens = nltk.word_tokenize(text)
                    for token in tokens:
                        vocab.add(token)
    vocab = list(vocab)
    vocab = sorted(vocab)
    vocab.insert(0, "<pad>")
    if require_unk:
        vocab.insert(1, "<unk>")
    print("vocab size: %d"%(len(vocab)))

    return vocab

def get_labels_from_corpus(data_dir, item_key="label"):
    labels = set()
    for file_name in ["train.json", "dev.json", "test.json"]:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    label = sample[item_key]
                    labels.add(label)
    labels = list(labels)
    labels = sorted(labels)
    print("label size: %d" % (len(labels)))
    return labels

def reverse_padded_sequence(inputs, lengths, batch_first=True):
    """
    Reverses sequences according to their lengths.
    B: batch_size, T: seq_length,
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Args:
        inputs: padded batch of variable length sequences.
        lengths: list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")
    ind = [list(reversed(range(0, length))) + list(range(length, max_length)) for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda()
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)

    return reversed_inputs

def build_embedding_of_corpus(embed_file, vocab, embed_dim, data_dir):
    """
    load embedding from embed_file
    Args:
        embed_file: pretrained embedding file
        vocab: a list of words
        embed_dim:
    Returns:
        corpus_embed:
    """
    processed_embed_name = "processed_embed.pkl"
    processed_file = os.path.join(data_dir, processed_embed_name)
    if os.path.exists(processed_file):
        with open(processed_file, "rb") as f:
            corpus_embed = pickle.load(f)
    else:
        word2vec = {}
        with open(embed_file, "r", encoding="utf-8") as f:
            idx = 0
            load_word_num = 0
            for line in f:
                line = line.strip()
                if line:
                    item = line.split()
                    if len(item) != (embed_dim+1):
                        continue
                    word = item[0]
                    vector = item[1:]
                    if word in vocab:
                        word2vec[word] = np.array(vector, dtype=np.float)
                        load_word_num += 1
                        if len(vocab) == load_word_num:
                            print("load embedding finished!")
                            break
                    idx += 1
                    # if idx > 100000:
                    #     break
                    if idx % 21800 == 0:
                        print("loading per%d"%(idx / 21800))

        corpus_embed = np.empty([len(vocab), embed_dim])
        scale = np.sqrt(3.0 / embed_dim)
        num_matched = 0
        num_non_matched = 0
        missing_words = []
        for idx, word in enumerate(vocab):
            if word in word2vec:
                corpus_embed[idx, :] = word2vec[word]
                num_matched += 1
            elif word.lower() in word2vec:
                corpus_embed[idx, :] = word2vec[word.lower()]
                num_matched += 1
            else:
                corpus_embed[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
                num_non_matched += 1
                missing_words.append(word)
        print("total: %d, matched: %d, non-matched: %d"%(len(vocab), num_matched, num_non_matched))
        print(missing_words[:10])
        with open(processed_file, "wb") as f:
            pickle.dump(corpus_embed, f)

    return corpus_embed
