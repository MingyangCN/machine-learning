import argparse
import os
import torch


def get_args_parser():
    """
        init parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base
    parser.add_argument("--dateset", default="cifar10", help="choose the dataset")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=1000, type=int)

    # args = parser.parse_args()
    return parser
