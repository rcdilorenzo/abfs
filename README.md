# ABFS - Automatic Building Footprint Segmentation

Videos: [Project Overview](https://www.youtube.com/watch?v=weYqdY7JY_g), [Liquid Cooling Upgrade](https://www.youtube.com/watch?v=Qdmf7I5BPtk)

Article: [Data Science from Concept to Production: A Case Study of ABFS](https://rcd.ai/abfs-project)

![ABFS hero banner](https://user-images.githubusercontent.com/634167/53965119-92100c80-40be-11e9-9568-e509db7b4870.jpg)

## Installation

<details>
  <summary>Installation Instructions</summary>
  
For this project, we'll use python 3.6.8. Go ahead and install `pyenv` if you don't already have it.

```
# * Install pyenv: https://github.com/pyenv/pyenv-installer
# * Add init commands to bash profile (bashrc, etc.)
# * Source shell before continuing

# Install proper version
pyenv install 3.6.8
```

Within the project directory, go ahead and setup a new virtual environment.

```
pyenv virtualenv 3.6.8 abfs-env
pyenv activate abfs-env
```

For `GDAL`, you'll need to install it separately through Homebrew/APT before installing the remaining requirements.

```
# Numpy must be installed BEFORE gdal (https://gis.stackexchange.com/a/274328)
pip install numpy

# On macOS:
brew install gdal

# On Debian/Ubuntu:
sudo apt-get install libgdal-dev
pip install \
  --global-option=build_ext \
  --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`
```

Now, go ahead and install the remaining dependencies.

```
pip install -r requirements.txt
```

For this program, you'll also need to decide whether to use a GPU-based backend.

```
# With CUDA-based GPU:
pip install tensorflow-gpu

# Without GPU:
pip install tensorflow
```

With these packages now available, install the command line utility.

```
python setup.py install
```

Verify it is installed properly by running the CLI.

```
abfs
```

If this returns an error about the command not being found, you may have to prepend the current python binary.

```
python -m abfs <COMMAND> <OPTIONS>
```
</details>


## Usage

The entire program is operated from the command line utility. Here are some examples.

```
❯ abfs -h
usage: abfs [-h] {serve,train,tune,export,evaluate} ...

positional arguments:
  {serve,train,tune,export,evaluate}
    serve               Serve model as API
    train               Train a neural network
    tune                Tune the tolerance parameter on the validation set
    export              Export keras model
    evaluate            Evaluate keras model based on test data

optional arguments:
  -h, --help            show this help message and exit
```

<details>
  <summary>Train</summary>

```
❯ abfs train -h
Using TensorFlow backend.
usage: abfs train [-h] [-lr LEARNING_RATE] [-s SIZE] [-e EPOCHS]
                  [-b BATCH_SIZE] [-mb MAX_BATCHES] [-gpus GPU_COUNT]

optional arguments:
  -h, --help            show this help message and exit
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
  -s SIZE, --size SIZE  Size of image
  -e EPOCHS, --epochs EPOCHS
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Number of examples per batch
  -mb MAX_BATCHES, --max-batches MAX_BATCHES
                        Maximum batches per epoch
  -gpus GPU_COUNT, --gpu-count GPU_COUNT

❯ abfs train -lr 0.02 --batch-size 8 --epochs 150 -gpus 2
...
```
</details>


<details>
  <summary>Tune (Includes Graph Output)</summary>

```
❯ abfs tune -h
usage: abfs tune [-h] [-w WEIGHTS_PATH] [-e MAX_EXAMPLES] [-s SIZE]
                 [-gpus GPU_COUNT]

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHTS_PATH, --weights-path WEIGHTS_PATH
                        Path to hdf5 model weights
  -e MAX_EXAMPLES, --max-examples MAX_EXAMPLES
                        Max number of examples to validate against
  -s SIZE, --size SIZE  Size of image
  -gpus GPU_COUNT, --gpu-count GPU_COUNT

❯ abfs tune -w checkpoints/<INSERT WEIGHTS HERE>.hdf5 -gpus 2
Tuning of the tolerance parameter will occur on 431 images.
Loading weights from checkpoints/<INSERT WEIGHTS HERE>.hdf5
Calculating F1-Scores... This may take perhaps even an hour if no GPU.
F1-Score calculation complete: 7.31 seconds
Plot has been saved to /tmp/tmpsyo8l1vi.png. Please open to view.
Tuned tolerance: 0.70 w/ median=0.6974 stdev=0.1722
```

![F1-Score Tuning Results](https://user-images.githubusercontent.com/634167/53690640-83cb9480-3d3c-11e9-99e6-e0efd910e22f.png)
</details>
  
<details>
  <summary>Export</summary>

```
❯ abfs export -h
Using TensorFlow backend.
usage: abfs export [-h] [-s SIZE] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  Size of image
  -o OUTPUT, --output OUTPUT

❯ abfs export -s 512 -o architecture
Using TensorFlow backend.
Save to "models/architecture.json"
```
</details>

<details>
  <summary>Evaluate</summary>

```
❯ abfs evaluate -h
Using TensorFlow backend.
usage: abfs evaluate [-h] [-w WEIGHTS_PATH] [-b BATCH_SIZE] [-s SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHTS_PATH, --weights-path WEIGHTS_PATH
                        Path to hdf5 model weights
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Number of examples per batch
  -s SIZE, --size SIZE  Size of image

❯ abfs evaluate -w checkpoints/unet-d82jd2-0020-0.19.hdf5
...
Loading weights from "checkpoints/unet-d82jd2-0020-0.19.hdf5"
Results:
[('loss', 0.1882165691484708), ...
```
</details>

<details>
  <summary>Serve</summary>

```
❯ abfs serve -h
Using TensorFlow backend.
usage: abfs serve [-h] [-w WEIGHTS_PATH] [-m MODEL_PATH] [-a ADDRESS]
                  [-p PORT]

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHTS_PATH, --weights-path WEIGHTS_PATH
                        Path to hdf5 model weights
  -m MODEL_PATH, --model-path MODEL_PATH
                        Path to keras model JSON
  -a ADDRESS, --address ADDRESS
                        Address to bind server to
  -p PORT, --port PORT  Port for server to listen on

❯ abfs serve \
    --weights-path checkpoints/unet-d82jd2-0020-0.19.hdf5 \
    --model-path models/unet-d82jd2.json \
    --mapbox-api-key <INSERT KEY HERE>
Using TensorFlow backend.
Serving on 0.0.0.0:1337
```
</details>

## Author

R. Christian Di Lorenzo. MIT License. ([About Me](https://rcd.ai/about-me))
