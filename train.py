#!/usr/bin/env python3

import torch
from transformers import BertModel, BertTokenizer
from argparse import ArgumentParser
from pathlib import Path
from csv import DictReader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
from itertools import zip_longest
import pickle
from copy import deepcopy
import numpy as np

BERT_OUTPUT_DIM = 768

class SurveyClassifier(torch.nn.Module):
  def __init__(self):
    super(SurveyClassifier, self).__init__()
    # Must transform a dense layer of size BERT_OUTPUT_DIM to
    # TrainingData.labels (relevant, activation, sentiment)

    # input
    self.l1 = torch.nn.Linear(BERT_OUTPUT_DIM, 3)
    # self.r1 = torch.nn.ReLU(inplace=True)
    # # batch 1
    # self.l2 = torch.nn.Linear(128, 3)

    # Order matters
    self.ops = [
        self.l1,
        # self.r1,
        # self.l2
    ]

  def forward(self, x):
    for op in self.ops:
      x = op(x)
    return x

################################################################################

def bert_to_sentence_embeddings(bert_model, tokenizer, sequences):
  return bert_model(sequences)[-2].mean(axis=1)

  # bad_tokens = [
      # tokenizer.pad_token_id,
      # # tokenizer.unk_token_id,
      # # tokenizer.sep_token_id,
      # # tokenizer.cls_token_id,
      # # tokenizer.mask_token_id,
  # ]

  # embedding = bert_model(sequences)[-2]
  # mask = torch.ones(sequences.shape, dtype=bool, device=sequences.device)
  # for bad_tok in bad_tokens:
    # mask &= (sequences != bad_tok)
  # mask = mask.unsqueeze(-1).expand_as(embedding)
  # embedding *= mask
  # return embedding.sum(axis=1) / mask.sum(axis=1)


def normalize_to_zero_one(one_six):
  assert 1 <= one_six <= 6
  return (one_six - 1) / 5.0

def denormalize_to_one_six(zero_one):
  zero_one = max(0, min(1, zero_one))
  return int(np.round(zero_one * 5 + 1))

class TrainingData(object):
  def __init__(self, text, relevant, activation, sentiment):
    self.text = text
    self.relevant = float(relevant) if relevant != "" else None
    self.activation = float(activation) if activation != "" else None
    self.sentiment = float(sentiment) if sentiment != "" else None
    self.embedding = None
    self.labels = None
    if self.is_complete():
      self.labels = torch.FloatTensor([
        self.relevant,
        normalize_to_zero_one(self.activation),
        normalize_to_zero_one(self.sentiment),
      ])

  def set_embedding(self, embedding):
    self.embedding = torch.FloatTensor(embedding)

  def get_embedding(self):
    return self.embedding

  def is_complete(self):
    return (
        self.relevant is not None
        and self.activation is not None
        and self.sentiment is not None
    )

  def get_labels(self):
    return self.labels


def iter_to_batches(iterable, batch_size):
  args = [iter(iterable)] * batch_size
  for batch in zip_longest(*args):
    yield list(filter(lambda b: b is not None, batch))

def get_args():
  parser = ArgumentParser()
  parser.add_argument("--pretrained-weights", default="bert-base-uncased")
  parser.add_argument("--raw-data", type=Path, default=Path("./data.csv"))
  parser.add_argument(
      "--processed-data",
      type=Path,
      default=Path("./processed_data.pkl")
  )
  parser.add_argument("--model", type=Path, default=Path("./model.pt"))
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--disable-gpu", action="store_true")
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--max-sequence_length", type=int, default=400)
  parser.add_argument("--test-ratio", type=float, default=0.2)
  parser.add_argument("--validation-ratio", type=float, default=0.1)
  parser.add_argument("--learning-rate", type=float, default=0.02)
  return parser.parse_args()

################################################################################

if __name__ == "__main__":
  args = get_args()

  print("Configuring pytorch")
  if torch.cuda.is_available() and not args.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
    torch.cuda.empty_cache()

  if args.processed_data.is_file():
    print(f"Loading previously-processed data from {args.processed_data}")
    with open(args.processed_data, 'rb') as data_file:
      training_data, validation_data, testing_data = pickle.load(data_file)
  else:
    print("Preprocessing all data")
    all_data = []
    assert args.raw_data.is_file()
    with open(args.raw_data) as csv_file:
      reader = DictReader(csv_file)
      for row in reader:
        all_data.append(
            TrainingData(
              text=row["text"],
              relevant=row["relevant"],
              activation=row["activation"],
              sentiment=row["sentiment"],
            )
        )
    print(f"Loaded {len(all_data)} values from {args.raw_data}")

    print(f"Filtering to only labeled data")
    all_data = list(filter(lambda x: x.is_complete(), all_data))
    print(f"Number of labeled points: {len(all_data)}")

    print(
        f"Loading {args.pretrained_weights}. This will download weights the "
        "first time."
    )
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_weights)
    embedding_model = BertModel.from_pretrained(args.pretrained_weights)
    embedding_model.eval()

    print(f"Sending model to compute device: {device}")
    embedding_model = embedding_model.to(device)

    print("Performing initial embedding")
    for batch in tqdm(
        iter_to_batches(all_data, args.batch_size),
        total=int(len(all_data)/args.batch_size)
    ):
      sequences = pad_sequence(
          sequences=[
            torch.tensor(
              tokenizer.encode(
                b.text,
                add_special_tokens=True
              )[:args.max_sequence_length]
            )
            for b in batch
          ],
          batch_first=True,
      ).to(device)
      embeddings = (
          bert_to_sentence_embeddings(
            embedding_model,
            tokenizer,
            sequences
          )
          .cpu()
          .detach()
          .numpy()
      )
      for element, embedding in zip(batch, embeddings):
        element.set_embedding(embedding)

    del embedding_model

    print("Shuffling")
    shuffle(all_data)

    print("Splitting to train - validation - test")
    training_data, testing_data = train_test_split(
        all_data,
        test_size=args.test_ratio
    )
    training_data, validation_data = train_test_split(
        training_data,
        test_size=args.validation_ratio
    )

    print(f"Saving training/validation/testing data to {args.processed_data}")
    with open(args.processed_data, 'wb') as data_file:
      pickle.dump([training_data, validation_data, testing_data], data_file)

  print(
      f"Split sizes: {len(training_data)}"
      f" - {len(validation_data)}"
      f" - {len(testing_data)}"
  )

  ##############################################################################

  print("Training specialty model")
  model = SurveyClassifier().to(device)
  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=args.learning_rate,
  )

  best_validation_loss = None
  best_model = None

  for epoch in range(args.epochs):
    shuffle(training_data)
    for phase in ["train", "validate", "test"]:
      if phase == "train":
        model.train()
        data = training_data
      elif phase == "validate":
        model.eval()
        data = validation_data
      elif phase == "test":
        if epoch != args.epochs-1:
          continue
        assert best_model is not None
        model = best_model
        model.eval()
        data = testing_data

      # Defining a progress bar link this so we can alter the description
      pbar = tqdm(
          iter_to_batches(data, args.batch_size),
          total=int(len(data)/float(args.batch_size)),
      )
      running_loss = 0.0
      running_count = 0.0
      for batch in pbar:
        batch_embeddings = (
            torch.stack([x.get_embedding() for x in batch])
            .to(device)
        )
        batch_labels = (
            torch.stack([x.get_labels() for x in batch])
            .to(device)
        )
        batch_predictions = model(batch_embeddings)
        batch_loss = loss_fn(batch_predictions, batch_labels)

        if phase == "train":
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()

        running_loss += batch_loss.detach() * len(batch)
        running_count += len(batch)
        pbar.set_description(f"{phase} -- Loss:{running_loss / running_count}")
      if phase == "validate":
        epoch_loss = running_loss / running_count
        if best_validation_loss is None or epoch_loss < best_validation_loss:
          best_validation_loss = epoch_loss
          best_model = deepcopy(model)
          print("Updating best model")

  print(f"Saving model to {args.model}")
  torch.save(best_model.state_dict(), args.model)
