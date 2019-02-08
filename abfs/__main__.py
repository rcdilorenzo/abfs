import sys
import inspect
import argparse
from toolz import first

from abfs.keras.u_net import UNet
from abfs.data import Data
from abfs.constants import *
from abfs.group_data_split import DataSplitConfig


def train(args):
    # Use Rio 3band data
    config = DataConfig(DEFAULT_DATA_DIR, BAND3, RIO_REGION)

    # Train (60%), Val (10%), Test (20%), Fixed seed
    split_config = DataSplitConfig(0.1, 0.2, 1337)

    # Create data with batch size
    data = Data(config, split_config, batch_size=args.batch_size, augment=True)

    # Exclude empty polygons
    data.data_filter = lambda df: df.sq_ft > 0

    # Create model
    unet = UNet(data, (args.size, args.size),
                max_batches=args.max_batches,
                epochs=args.epochs,
                gpu_count=args.gpu_count,
                learning_rate=args.learning_rate)

    # Print summary
    unet.model.summary()

    # Begin training
    unet.train()


def function_named(name):
    functions = inspect.getmembers(
        sys.modules[__name__],
        predicate=(lambda f: inspect.isfunction(f) and f.__module__ == __name__)
    )
    return first([v for k, v in functions if k == name])

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help='Train a neural network')
    train_parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    train_parser.add_argument('-s', '--size', type=int, default=512,
                              help='Size of image')
    train_parser.add_argument('-e', '--epochs', type=int, default=100)
    train_parser.add_argument('-b', '--batch-size', type=int, default=4,
                              help='Number of examples per batch')
    train_parser.add_argument('-mb', '--max-batches', type=int, default=999999,
                              help='Maximum batches per epoch')
    train_parser.add_argument('-gpus', '--gpu-count', type=int, default=1)
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
