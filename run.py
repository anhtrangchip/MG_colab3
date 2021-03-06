import os, time, datetime
import torch
import torch.nn as nn
from random import shuffle
from preprocess import PreprocessingPipeline
from train import train
from model import MusicTransformer
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser("Script to train model on a GPU")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional path to saved model, if none provided, the model is trained from scratch.")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--ckpt_number", type=int, default=1,
                        help="version of checkpoint directory")
    args = parser.parse_args()

    sampling_rate = 125
    n_velocity_bins = 32
    seq_length = 1024
    n_tokens = 256 + sampling_rate + n_velocity_bins
    transformer = MusicTransformer(n_tokens, seq_length,
                                   d_model=64, n_heads=8, d_feedforward=256,
                                   depth=4, positional_encoding=False, relative_pos=True)

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint)
        transformer.load_state_dict(state)
        print(f"Successfully loaded checkpoint at {args.checkpoint}")
    else:
        print(f"NOT FOUND checkpoint")

    pipeline = PreprocessingPipeline(input_dir="data", split_size=30, sampling_rate=sampling_rate,
                                     n_velocity_bins=n_velocity_bins, training_val_split=0.9,
                                     max_encoded_length=seq_length + 1, min_encoded_length=257,
                                     stretch_factors = [0.95, 1, 1.1])
    pipeline_start = time.time()
    pipeline.run()
    runtime = time.time() - pipeline_start
    print(f"MIDI pipeline runtime: {runtime / 60 : .1f}m")

    today = datetime.date.today().strftime('%m%d%Y')
    checkpoint = f"drive/MyDrive/UETK62/saved_models{args.ckpt_number}/tf_{today}"
    print(checkpoint)

    training_sequences = pipeline.encoded_sequences['training']
    validation_sequences = pipeline.encoded_sequences['validation']

    batch_size = 1
    print("batch size: ", batch_size)

    train(transformer, training_sequences, validation_sequences,
          epochs=args.n_epochs, evaluate_per=1,
          batch_size=batch_size, batches_per_print=100,
          padding_index=0, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()