import sys
import inspect
import argparse
from toolz import pipe

from abfs.cli import *

def add_train_command(subparsers):
    parser = subparsers.add_parser('train', help='Train a neural network')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-s', '--size', type=int, default=512,
                        help='Size of image')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='Number of examples per batch')
    parser.add_argument('-mb', '--max-batches', type=int, default=999999,
                        help='Maximum batches per epoch')
    parser.add_argument('-wp', '--weights-path', type=str, default=None,
                        help='Path to existing weights path')
    parser.add_argument('-gpus', '--gpu-count', type=int, default=1)
    parser.set_defaults(func=train)
    return subparsers

def add_export_command(subparsers):
    parser = subparsers.add_parser('export', help='Export keras model')
    parser.add_argument('-s', '--size', type=int, default=512,
                        help='Size of image')
    parser.add_argument('-o', '--output', type=str, default='model_output')
    parser.set_defaults(
        func=export, epochs=None, gpu_count=1, learning_rate=0,
        max_batches=9999999, batch_size=0
    )
    return subparsers

def add_evaluate_command(subparsers):
    parser = subparsers.add_parser('evaluate',
                                   help='Evaluate keras model based on test data')
    parser.add_argument('-w', '--weights-path', type=str, default=None,
                        help='Path to hdf5 model weights')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='Number of examples per batch')
    parser.add_argument('-s', '--size', type=int, default=512,
                        help='Size of image')
    parser.set_defaults(
        func=evaluate, epochs=None, gpu_count=1, learning_rate=0,
        max_batches=9999999
    )
    return subparsers


def add_tune_command(subparsers):
    parser = subparsers.add_parser(
        'tune',
        help='Tune the tolerance parameter on the validation set'
    )
    parser.add_argument('-w', '--weights-path', type=str,
                        help='Path to hdf5 model weights')
    parser.add_argument('-e', '--max-examples', type=int, default=999998,
                        help='Max number of examples to validate against')
    parser.add_argument('-s', '--size', type=int, default=512,
                        help='Size of image')
    parser.add_argument('-gpus', '--gpu-count', type=int, default=1)
    parser.set_defaults(
        func=tune, epochs=None, learning_rate=0,
        max_batches=9999999
    )
    return subparsers

def add_serve_command(subparsers):
    parser = subparsers.add_parser('serve', help='Serve model as API')
    parser.add_argument('-w', '--weights-path', type=str,
                        help='Path to hdf5 model weights')
    parser.add_argument('-ws3', '--weights-s3', type=str,
                        help='S3 path to hdf5 model weights')

    parser.add_argument('-m', '--model-path', type=str,
                        help='Path to keras model JSON')
    parser.add_argument('-ms3', '--model-s3', type=str,
                        help='S3 path to keras model JSON')

    parser.add_argument('-a', '--address', type=str, default='0.0.0.0',
                        help='Address to bind server to')
    parser.add_argument('-p', '--port', type=int, default=1337,
                        help='Port for server to listen on')
    parser.add_argument('-mb', '--mapbox-api-key', type=str, default='',
                        help='Mapbox API key')
    parser.set_defaults(func=serve)
    return subparsers


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    pipe(
        parser.add_subparsers(),
        add_serve_command,
        add_train_command,
        add_tune_command,
        add_export_command,
        add_evaluate_command
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
