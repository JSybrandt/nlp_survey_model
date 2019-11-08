#!/usr/bin/env python3
from train import(
    SurveyClassifier,
    TrainingData,
    iter_to_batches,
    get_args,
    denormalize_to_one_six,
    bert_to_sentence_embeddings,
)
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import sys
import numpy as np
from csv import DictReader, DictWriter

if __name__ == "__main__":
  args = get_args()

  if torch.cuda.is_available() and not args.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  tokenizer = BertTokenizer.from_pretrained(args.pretrained_weights)
  survey_model = SurveyClassifier.from_pretrained(args.pretrained_weights)
  survey_model.load_state_dict(torch.load(args.model))
  survey_model = survey_model.eval().to(device)

  for data_path in args.raw_data:
    assert data_path.is_file()
    with open(data_path) as csv_file:
      reader = DictReader(csv_file)
      fields = [
          "text",
          "sentiment",
          "activation",
          "predicted_sentiment",
          "predicted_activation",
      ]
      writer = DictWriter(sys.stdout, fieldnames=fields)
      writer.writeheader()
      for row in reader:
        line = row["text"].strip().lower()
        sequences = pad_sequence(
          sequences=[
            torch.tensor(
              tokenizer.encode(
                line,
                add_special_tokens=True,
                max_length=args.max_sequence_length,
              )
            )
          ],
          batch_first=True,
        ).to(device)
        activation, sentiment = (
            survey_model(sequences).cpu().detach().tolist()[0]
        )
        activation = denormalize_to_one_six(activation)
        sentiment = denormalize_to_one_six(sentiment)
        writer.writerow({
          "text": row["text"],
          "sentiment": row["sentiment"] if "sentiment" in row else "N/A",
          "activation": row["activation"] if "activation" in row else "N/A",
          "predicted_sentiment": sentiment,
          "predicted_activation": activation,
        })
