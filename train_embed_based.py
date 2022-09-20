# author = liuwei
# date = 2022-09-20

import os
import json
import random
import time
import datetime
from tqdm import tqdm, trange

import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from task_dataset import EmbedDataset
from models import TextLSTM
from utils import get_vocab_from_corpus, get_labels_from_corpus

# for output
dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"

def get_argparse():
    parser = argparse.ArgumentParser()

    # for data
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="test", type=str, help="toefl1234, gcdc")
    parser.add_argument("--fold_id", default=-1, type=int, help="[1-5] for toefl1234, [1-10] for gcdc")
    parser.add_argument("--output_dir", default="data/result", type=str, help="path to save checkpoint")
    parser.add_argument("--embed_file", default="data/embedding/glove.840B.300d.txt", type=str)
    parser.add_argument("--text_key", default="text", type=str)
    parser.add_argument("--label_key", default="score", type=str)

    # for model
    parser.add_argument("--model_type", default="textlstm", type=str, help="models for text classification")
    parser.add_argument("--input_dim", default=300, type=int, help="the dimension size of fc layer")
    parser.add_argument("--hidden_size", default=240, type=int, help="the dimension size of fc layer")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, action="store_true", help="Whether to do evaluation")
    parser.add_argument("--do_pred", default=False, action="store_true")
    parser.add_argument("--max_seq_length", default=64, type=int, help="the max length of input sequence")
    parser.add_argument("--train_batch_size", default=4, type=int, help="the training batch size")
    parser.add_argument("--eval_batch_size", default=36, type=int, help="the eval batch size")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="training epoch, only work when max_step==-1")
    parser.add_argument("--learning_rate", default=5e-2, type=float, help="The initial learning rate for Adam")
    parser.add_argument("--dropout", default=0.5, type=float, help="dropout value")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="the weight of L2 normalization")
    parser.add_argument("--warmup_ratio", default=0.0, type=float, help="warmup rate duing train")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="max gradient value")
    parser.add_argument("--seed", default=106524, type=int, help="the seed used to initiate parameters")

    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset(args, mode="train"):
    if args.model_type.lower() == "textlstm":
        data_params = {
            "max_seq_length": args.max_seq_length,
            "labels": args.labels,
            "vocab": args.vocab,
            "text_key": args.text_key,
            "label_key": args.label_key
        }
        file_name = os.path.join(args.data_dir, "{}.json".format(mode))
        dataset = EmbedDataset(file_name, data_params)

    return dataset

def get_dataloader(dataset, args, mode="train"):
    if mode.upper() == 'TRAIN':
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader

def get_model(args):
    if args.model_type.lower() == "textlstm":
        params = {
            "vocab": args.vocab,
            "embed_file": args.embed_file,
            "data_dir": args.data_dir,
            "input_dim": args.input_dim,
            "hidden_size": args.hidden_size,
            "num_layers": 1,
            "num_labels": len(args.labels),
            "dropout": args.dropout
        }
        model = TextLSTM(params)

    return model

def get_optimizer(model, args, num_training_steps):
    no_deday = []
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_deday)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_deday)],
            "weight_decay": 0.0
        }
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    return optimizer, scheduler

def train(model, args, train_dataset, dev_dataset, test_dataset):
    train_dataloader = get_dataloader(train_dataset, args, mode="train")
    num_train_epochs = args.num_train_epochs
    num_train_steps = int(len(train_dataloader) * num_train_epochs)
    optimizer, scheduler = get_optimizer(model, args, num_train_steps)

    global_step = 0
    avg_loss = 0.0
    cur_loss = 0.0
    best_dev = 0.0
    best_dev_epoch = 0
    best_test = 0.0
    best_test_epoch = 0
    model.zero_grad()
    train_iter = trange(1, num_train_epochs + 1, desc="Epoch")
    for epoch in train_iter:
        epoch_iter = tqdm(train_dataloader, desc="Iteration")
        model.train()
        for step, batch in enumerate(epoch_iter):
            optimizer.zero_grad()
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type.lower() == "textlstm":
                inputs = {"input_ids": batch[0], "seq_lengths": batch[1], "labels": batch[2], "flag": "Train"}
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            cur_loss = loss.item()
            avg_loss += cur_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            # scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                print("global step: %d, cur loss: %.2f, avg loss: %.2f"%(global_step, cur_loss, avg_loss / global_step))
        # update lr
        scheduler.step()

        # evaluation after each epoch
        model.eval()
        train_acc, train_f1 = evaluate(model, args, train_dataset, desc="train")
        dev_acc, dev_f1 = evaluate(model, args, dev_dataset, desc="dev")
        test_acc, test_f1 = evaluate(model, args, test_dataset, desc="test")
        print("Epoch=%d, Train: Acc=%.4f, F1=%.4f" % (epoch, train_acc, train_f1))
        print("Epoch=%d, Dev: Acc=%.4f, F1=%.4f" % (epoch, dev_acc, dev_f1))
        print("Epoch=%d, Test: Acc=%.4f, F1=%.4f" % (epoch, test_acc, test_f1))
        if (dev_acc > best_dev) or (dev_acc == best_dev and test_acc > best_test):
            best_dev = dev_acc
            best_dev_epoch = epoch
        if test_acc > best_test:
            best_test = test_acc
            best_test_epoch = epoch
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print("Best dev: Epoch=%d, Acc=%.4f" % (best_dev_epoch, best_dev))
    print("Best test: Epoch=%d, Acc=%.4f" % (best_test_epoch, best_test))

def evaluate(model, args, dataset, desc="dev", write_file=False):
    dataloader = get_dataloader(dataset, args, mode=desc)
    all_label_ids = None
    all_pred_ids = None
    label_index = -1
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        if args.model_type.lower() == "textlstm":
            inputs = {"input_ids": batch[0], "seq_lengths": batch[1], "labels": batch[2], "flag": "Eval"}
            label_index = 2
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        label_ids = batch[label_index].detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        
        if all_label_ids is None:
            all_label_ids = label_ids
            all_pred_ids = preds
        else:
            all_label_ids = np.append(all_label_ids, label_ids)
            all_pred_ids = np.append(all_pred_ids, preds)
    
    acc = accuracy_score(y_true=all_label_ids, y_pred=all_pred_ids)
    f1 = f1_score(y_true=all_label_ids, y_pred=all_pred_ids, average="macro")

    if write_file:
        res_file = "{}_res_{}.txt".format(desc, args.model_type)
        data_dir = os.path.join(args.data_dir, "preds")
        os.makedirs(data_dir, exist_ok=True)
        res_file = os.path.join(data_dir, res_file)
        error_num = 0
        with open(res_file, "w", encoding="utf-8") as f:
            for l, p in zip(all_label_ids, all_pred_ids):
                if l == p:
                    f.write("%s\t%s\n" % (l, p))
                else:
                    error_num += 1
                    f.write("%s\t%s\t%d\n" % (l, p, error_num))

    return acc, f1

def main():
    args = get_argparse().parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args.device = device
    print("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    # 1. data related
    data_dir = os.path.join(args.data_dir, args.dataset)
    if args.fold_id > 0:
        data_dir = os.path.join(data_dir, str(args.fold_id))
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, args.model_type)
    if args.fold_id > 0:
        output_dir = os.path.join(output_dir, str(args.fold_id))
    args.output_dir = output_dir
    labels = get_labels_from_corpus(args.data_dir, item_key="score")
    vocab = get_vocab_from_corpus(args.data_dir)
    args.labels = labels
    args.vocab = vocab

    # 2. model
    model = get_model(args)
    model = model.to(args.device)

    # 3. train and evaluation
    if args.do_train:
        print("Acc for {} fold {}: ".format(args.dataset, args.fold_id))
        train_dataset = get_dataset(args, mode="train")
        dev_dataset = get_dataset(args, mode="dev")
        test_dataset = get_dataset(args, mode="test")
        print("Train size: %d, Dev size: %d, Test size: %d"%(train_dataset.total_size, dev_dataset.total_size, test_dataset.total_size))
        train(model, args, train_dataset, dev_dataset, test_dataset)

    if args.do_eval or args.do_pred:
        checkpoint_file = "data/result/"
        model.load_state_dict(torch.load(checkpoint_file))
        model.eval()

        if args.do_eval:
            dataset = get_dataset(args, mode="dev")
            acc, f1 = evaluate(model, args, dataset, desc="dev", write_file=True)
            print("Dev Acc=%.4f, F1: %.4f" % (acc, f1))

        if args.do_pred:
            dataset = get_dataset(args, mode="test")
            acc, f1 = evaluate(model, args, dataset, desc="test", write_file=True)
            print("Test Acc=%.4f, F1: %.4f" % (acc, f1))

if __name__ == "__main__":
    main()
