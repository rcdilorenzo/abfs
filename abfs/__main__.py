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

    # Create data with batch size of 8
    data = Data(config, split_config, batch_size=4)

    # Exclude empty polygons
    data.data_filter = lambda df: df.sq_ft > 0

    # Create model
    unet = UNet(data, (512, 512), max_batches=40)

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
    parser.add_argument('command', type=str, choices=['train'])
    args = parser.parse_args()
    function_named(args.command)(args)

if __name__ == "__main__":
    main()
