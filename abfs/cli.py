from abfs.keras.u_net import UNet
from abfs.data import Data
from abfs.constants import *
from abfs.group_data_split import DataSplitConfig
from abfs.api.server import serve as api_serve

def model(args):
    # Use Rio 3band data
    config = DataConfig(DEFAULT_DATA_DIR, BAND3, RIO_REGION)

    # Train (60%), Val (10%), Test (20%), Fixed seed
    split_config = DataSplitConfig(0.1, 0.2, 1337) # Create data with batch size
    data = Data(config, split_config, batch_size=args.batch_size, augment=True)

    # Exclude empty polygons
    data.data_filter = lambda df: df.sq_ft > 0

    # Create model
    return UNet(data, (args.size, args.size),
                max_batches=args.max_batches,
                epochs=args.epochs,
                gpu_count=args.gpu_count,
                learning_rate=args.learning_rate)

def train(args):
    unet = model(args)
    unet.model.summary()
    unet.train()


def evaluate(args):
    print('Evaluating...')
    unet = model(args)

    unet.compile()

    print(f'Loading weights from "{args.weights_path}"')
    unet.model.load_weights(args.weights_path, by_name=False)

    result = unet.evaluate()
    print(f'Results: \n{list(zip(unet.model.metrics_names, result))}')


def export(args):
    unet = model(args)

    path = f'models/{args.output}.json'
    with open(path, 'w') as f:
        f.write(unet.model.to_json())

    print(f'Save to "{path}"')

def serve(args):
    api_serve(args.address, args.port, args.model_path,
              args.weights_path, args.mapbox_api_key)

