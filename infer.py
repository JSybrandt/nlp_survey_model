#!/usr/bin/env python3
from train import(
    SurveyClassifier,
    TrainingData,
    iter_to_batches,
    get_args,
    denormalize_to_one_six,
)
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import sys
import numpy as np

if __name__ == "__main__":
  args = get_args()

  print("Configuring pytorch")
  if torch.cuda.is_available() and not args.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  print(
      f"Loading {args.pretrained_weights}. This will download weights the "
      "first time."
  )
  tokenizer = BertTokenizer.from_pretrained(args.pretrained_weights)
  embedding_model = (
      BertModel
      .from_pretrained(args.pretrained_weights)
      .eval()
      .to(device)
  )

  print(f"Loading model from {args.model}")
  survey_model = SurveyClassifier()
  survey_model.load_state_dict(torch.load(args.model))
  survey_model = survey_model.eval().to(device)

  print("Press CTRL+D to stop reading. Assuming one text per line.")
  for line in sys.stdin:
    line = line.strip().lower()
    sequences = pad_sequence(
      sequences=[
        torch.tensor(tokenizer.encode(line)[:args.max_sequence_length])
      ],
      batch_first=True,
    ).to(device)
    embedding = embedding_model(sequences)[-1]
    relevant, activation, sentiment = (
        survey_model(embedding).cpu().detach().tolist()[0]
    )
    relevant = int(np.round(relevant))
    activation = denormalize_to_one_six(activation)
    sentiment = denormalize_to_one_six(sentiment)
    print(
        f"Relevant: {relevant}, "
        f"Activation: {activation}, "
        f"Sentiment: {sentiment}"
    )
