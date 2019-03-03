from abfs.keras.u_net import UNet
from abfs.data import Data
from abfs.constants import *
from abfs.group_data_split import DataSplitConfig
from abfs.api.server import serve as api_serve

def model(args):
    # Use Rio 3band data
    config = DataConfig(DEFAULT_DATA_DIR, BAND3, RIO_REGION)

    # Train (70%), Val (10%), Test (20%), Fixed seed
    split_config = DataSplitConfig(0.1, 0.2, 1337) # Create data with batch size
    data = Data(config, split_config,
                batch_size=args.max_examples if hasattr(args, 'max_examples') else args.batch_size,
                augment=True)

    # Exclude empty polygons
    data.data_filter = lambda df: df.sq_ft > 0

    # Create model
    return UNet(data, (args.size, args.size),
                max_batches=args.max_batches,
                epochs=args.epochs,
                weights_path=args.weights_path,
                gpu_count=args.gpu_count,
                learning_rate=args.learning_rate)

def train(args):
    unet = model(args)
    unet.model.summary()
    unet.compile()
    unet.train()

def tune(args):
    unet = model(args)
    unet.tune_tolerance()


def evaluate(args):
    print('Evaluating...')
    unet = model(args)

    result = unet.evaluate()
    print(f'Results: \n{list(zip(unet.model.metrics_names, result))}')


def export(args):
    unet = model(args)

    path = f'models/{args.output}.json'
    with open(path, 'w') as f:
        f.write(unet.model.to_json())

    print(f'Save to "{path}"')


def serve(args):
    api_serve(args.address, args.port,
              args.model_path or _download_s3_object(args.model_s3),
              args.weights_path or _download_s3_object(args.weights_s3),
              args.mapbox_api_key)


def _download_s3_object(s3_url):
    import boto3
    import tempfile

    if s3_url is None: return None
    bucket, key = s3_url.split('/',2)[-1].split('/',1)
    file_path = tempfile.NamedTemporaryFile('w+b', delete=False)
    print(f'Downloading {key}...')
    boto3.resource('s3').Bucket(bucket).download_file(key, file_path.name)
    print(f'Downloaded {key}.')
    return file_path.name

