#!/usr/bin/env python3

import torch
from transformers import BertModel, BertTokenizer
from typing import Dict, Any
from argparse import ArgumentParser
from pathlib import Path
from csv import DictReader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
from itertools import zip_longest
import pickle
from copy import deepcopy, copy
import numpy as np
import horovod.torch as hvd

BERT_OUTPUT_DIM = 768

class SurveyClassifier(BertModel):
  def __init__(self, config:Dict[str, Any], freeze_bert_layers=True):
    super(SurveyClassifier, self).__init__(config)
    for param in self.parameters():
      param.requires_grad = not freeze_bert_layers

    # input
    # self.drop = torch.nn.Dropout(p=0.2)
    self.l1 = torch.nn.Linear(BERT_OUTPUT_DIM, 2)
    self.r1 = torch.nn.ReLU(inplace=True)

    # Order matters
    self.ops = [
        # self.drop,
        self.l1,
        self.r1,
    ]

  def unfreeze_layers_starting_with(self, level:int):
    assert level >= 0
    assert level <= 11
    for name, param in self.named_parameters():
      tokens = name.split(".")
      if ((
            tokens[0] == "encoder"
            and tokens[1] == "layer"
            and int(tokens[2]) >= level
          )
          or (
            tokens[0] == "pooler"
            and tokens[1] == "dense"
          )
      ):
        param.requires_grad = True

  def forward(self, *args, **kwargs):
    x =  super(SurveyClassifier, self).forward(*args, **kwargs)[0]
    attention_mask = kwargs["attention_mask"].unsqueeze(2)
    assert x.shape[0] == attention_mask.shape[0]
    assert x.shape[1] == attention_mask.shape[1]
    assert attention_mask.shape[2] == 1
    x *= attention_mask
    x = x.sum(dim=1) / attention_mask.sum(dim=1)
    for op in self.ops:
      x = op(x)
    return x

################################################################################

def binary_accuracy(prediction, labels):
  num_correct = ((prediction > 0.5) == (labels > 0.5)).sum().float()
  return num_correct / prediction.numel()

def per_class_accuracy(prediction, labels):
  def denorm(val):
    val = copy(val).detach()
    val[val<0]=0
    val[val>1]=1
    val *= 5
    val += 1
    return val.round()
  return (denorm(prediction) == denorm(labels)).sum().float() / prediction.numel()


def bert_to_sentence_embeddings(bert_model, tokenizer, sequences):
  return bert_model(sequences)[0].mean(axis=1)


def normalize_to_zero_one(one_six):
  assert 1 <= one_six <= 6
  return (one_six - 1) / 5.0

def denormalize_to_one_six(zero_one):
  zero_one = max(0, min(1, zero_one))
  return zero_one * 5 + 1

class TrainingData(object):
  def __init__(self, text, activation, sentiment):
    self.text = text
    self.activation = float(activation) if activation != "" else None
    self.sentiment = float(sentiment) if sentiment != "" else None
    self.embedding = None
    self.labels = None
    if self.is_complete():
      self.labels = torch.FloatTensor([
        normalize_to_zero_one(self.activation),
        normalize_to_zero_one(self.sentiment),
      ])

  def is_complete(self):
    return (
        self.activation is not None
        and self.sentiment is not None
    )

  def get_labels(self):
    return self.labels


def iter_to_batches(iterable, batch_size):
  args = [iter(iterable)] * batch_size
  for batch in zip_longest(*args):
    yield list(filter(lambda b: b is not None, batch))

def split_by_rank(data):
  points_per_rank = int(len(data) / hvd.size())
  first = hvd.rank() * points_per_rank
  last = first + points_per_rank
  return data[first:last]

def get_args():
  parser = ArgumentParser()
  parser.add_argument("--pretrained-weights", default="bert-base-uncased")
  parser.add_argument("--raw-data", type=Path, nargs="+", default=[
    Path("./data/tiffany.csv"),
    Path("./data/justin.csv"),
    Path("./data/mike.csv"),
  ])
  parser.add_argument("--model", type=Path, default=Path("./model.pt"))
  parser.add_argument("--batch-size", type=int, default=20)
  parser.add_argument("--disable-gpu", action="store_true")
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--unfreeze-bert-epoch", type=int, default=5)
  parser.add_argument("--max-sequence_length", type=int, default=500)
  parser.add_argument("--validation-ratio", type=float, default=0.1)
  parser.add_argument("--learning-rate", type=float, default=0.002)
  return parser.parse_args()



################################################################################

if __name__ == "__main__":
  args = get_args()

  seed = 42
  hvd.init()

  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.set_num_threads(1)
  torch.cuda.set_device(hvd.local_rank())

  print("Configuring pytorch")
  if torch.cuda.is_available() and not args.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  print("Preprocessing all data")
  training_data = []
  for data_path in args.raw_data:
    assert data_path.is_file()
    with open(data_path) as csv_file:
      reader = DictReader(csv_file)
      for row in reader:
        training_data.append(
            TrainingData(
              text=row["text"],
              activation=row["activation"],
              sentiment=row["sentiment"],
            )
        )
  print(f"Loaded {len(training_data)} values from {args.raw_data}")

  print(f"Filtering to only labeled data")
  training_data = list(filter(lambda x: x.is_complete(), training_data))
  print(f"Number of labeled points: {len(training_data)}")

  training_data, validation_data = train_test_split(
      training_data,
      test_size=args.validation_ratio
  )
  training_data = split_by_rank(training_data)
  validation_data = split_by_rank(validation_data)


  tokenizer = BertTokenizer.from_pretrained(args.pretrained_weights)
  model = SurveyClassifier.from_pretrained(args.pretrained_weights)
  model.to(device)
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)

  print("Shuffling")
  shuffle(training_data)

  ##############################################################################

  print("Training specialty model")
  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=args.learning_rate * hvd.size(),
  )
  optimizer = hvd.DistributedOptimizer(
      optimizer,
      named_parameters=model.named_parameters(),
  )
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)


  best_validation_loss = None
  best_model = None
  for epoch in range(args.epochs):
    if epoch  == args.unfreeze_bert_epoch:
      model.unfreeze_layers_starting_with(11)
    if epoch  == args.unfreeze_bert_epoch*2:
      model.unfreeze_layers_starting_with(10)
    for phase in ["train", "validate"]:
      if phase == "train":
        model.train()
        shuffle(training_data)
        data = training_data
      elif phase == "validate":
        model.eval()
        data = validation_data

      # Defining a progress bar link this so we can alter the description
      running_loss = 0.0
      running_act_2acc = 0.0
      running_sent_2acc = 0.0
      running_act_6acc = 0.0
      running_sent_6acc = 0.0
      running_count = 0.0
      for batch in iter_to_batches(data, args.batch_size):
        batch_seqs = pad_sequence(
            sequences=[
              torch.tensor(
                tokenizer.encode(
                  b.text,
                  add_special_tokens=True,
                  max_length=args.max_sequence_length,
                )
              )
              for b in batch
            ],
            batch_first=True,
        ).to(device)
        batch_mask = torch.ones_like(batch_seqs)
        # Mask out the padding tokens
        batch_mask[batch_seqs==0] = 0
        batch_labels = torch.stack([x.get_labels() for x in batch]).to(device)
        batch_predictions = model(batch_seqs, attention_mask=batch_mask)
        batch_loss = loss_fn(batch_predictions, batch_labels)

        running_act_2acc += binary_accuracy(
            batch_predictions[:, 0],
            batch_labels[:, 0]
        )
        running_sent_2acc += binary_accuracy(
            batch_predictions[:, 1],
            batch_labels[:, 1]
        )
        running_act_6acc += per_class_accuracy(
            batch_predictions[:, 0],
            batch_labels[:, 0]
        )
        running_sent_6acc += per_class_accuracy(
            batch_predictions[:, 1],
            batch_labels[:, 1]
        )

        if phase == "train":
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()

        running_loss += batch_loss.detach()
        running_count += 1
        if hvd.rank() == 0:
          print(
              f"{epoch} - {phase}"
              f" -- Loss:{running_loss / running_count:0.3f}"
              f" -- Act2:{running_act_2acc / running_count:0.3f}"
              f" -- Act6:{running_act_6acc / running_count:0.3f}"
              f" -- Sen2:{running_sent_2acc / running_count:0.3f}"
              f" -- Sen6:{running_sent_6acc / running_count:0.3f}"
          )

      if phase == "validate":
        epoch_loss = running_loss / running_count
        epoch_loss = hvd.allreduce(epoch_loss, name=f"epoch_loss")
        if hvd.rank() == 0:
          print(epoch_loss)
        if best_validation_loss is None or epoch_loss < best_validation_loss:
          best_validation_loss = epoch_loss
          best_model = deepcopy(model)

  if hvd.rank() == 0:
    print(f"Saving model to {args.model}")
    torch.save(best_model.state_dict(), args.model)
