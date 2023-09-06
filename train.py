import torchvision
from torch import nn
import numpy as np
from argparse import ArgumentParser
from src.trainer import Trainer

def add_arguments(parser: ArgumentParser):
    parser.add_argument("--model", choices=["cnn", "cnn_small_decoder"], default="cnn", help="model architecture to use")
    parser.add_argument("--model_save_path", type=str, required=True, help="path to checkpoint directory")
    parser.add_argument("--cw", action="store_true", help="whether to use cw regularization")
    parser.add_argument("--cw_coeff", type=float, default=0.5, help="cw regularization coefficient, ignored if cw=False")
    parser.add_argument("--data_root", type=str, required=True, help="directory containing training data")
    parser.add_argument("--dataset", type=str, choices=["celeba", "cifar100", "unsplash"], default="unsplash")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--image_dim", type=int, default=64)
    parser.add_argument("--latent_channel", type=int, default=2)

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam"], help="optimizer type")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--content_loss", type=str, default="l2", choices=["l2", "l1"])
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate decay for step scheduler")
    parser.add_argument("--step_size", type=int, default=5, help="scheduler step size")

    parser.add_argument("--saving_interval", type=int, default=5)

def train():
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    train()