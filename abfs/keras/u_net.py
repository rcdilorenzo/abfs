import numpy as np
from funcy import rpartial
from toolz import memoize, pipe, curry, topk, second
from time import time

import h5py
import math
import uuid
import tempfile
import warnings

import abfs.keras.metrics
from abfs.keras.generator import Generator
from abfs.keras.custom_tensor_board import CustomTensorBoard

import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
import keras.backend as K

third = lambda value: value[2]

class UNet():
    def __init__(self, data, shape,
                 max_batches=math.inf,
                 gpu_count=1,
                 epochs=100,
                 weights_path=None,
                 learning_rate=0.001):

        self.uuid = uuid.uuid4().hex[0:6]
        self.data = data
        self.shape = shape
        self.gpu_count = gpu_count
        self.max_batches = max_batches
        self.epochs = epochs
        self.weights_path = weights_path
        self.learning_rate = learning_rate

    def compile(self):
        self.model.compile(optimizer=SGD(lr=self.learning_rate),
                           loss='binary_crossentropy',
                           metrics=[abfs.keras.metrics.fbeta_score])


    def train(self):
        self.compile()
        self.model.fit_generator(self.train_generator,
                                 validation_data=self.val_generator,
                                 epochs=self.epochs,
                                 callbacks=self._callbacks())

    def evaluate(self):
        self.compile()
        return self.model.evaluate_generator(self.test_generator)

    @property
    @memoize
    def train_generator(self):
        return self.data.train_generator(Generator, self.shape,
                                         max_batches=self.max_batches)

    @property
    @memoize
    def val_generator(self):
        return self.data.val_generator(Generator, self.shape,
                                       max_batches=self.max_batches)

    @property
    @memoize
    def test_generator(self):
        return self.data.test_generator(Generator, self.shape)


    @curry
    def conv_block(self, filters, kernel_size, last_layer):
        return pipe(
            last_layer,
            Conv2D(filters, kernel_size,
                   activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal'),
            Conv2D(filters, kernel_size,
                   activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal')
        )

    @curry
    def conv_up_block(self, filters, counterpart_layer, last_layer):
        up = pipe(
            last_layer,
            UpSampling2D(size = (2,2)),
            Conv2D(filters, 2,
                   activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal')
        )
        conv_up = pipe(
            concatenate([counterpart_layer, up], axis=3),
            Conv2D(filters, 3,
                   activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal'),
            Conv2D(filters, 3,
                   activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal')
        )
        return conv_up


    @property
    @memoize
    def model(self):
        height, width = self.shape

        inputs = Input((height, width, 3))
        conv1 = self.conv_block(64, 3, inputs)
        conv1_out = pipe(
            conv1,
            MaxPooling2D(pool_size=(2, 2))
        )

        conv2 = self.conv_block(128, 3, conv1_out)
        conv2_out = pipe(
            conv2,
            Dropout(0.25),
            MaxPooling2D(pool_size=(2, 2))
        )

        conv3 = self.conv_block(256, 3, conv2_out)
        conv3_out = pipe(
            conv3,
            Dropout(0.25),
            MaxPooling2D(pool_size=(2, 2))
        )

        conv4 = self.conv_block(512, 3, conv3_out)
        conv4_out = pipe(
            conv4,
            Dropout(0.25),
            MaxPooling2D(pool_size=(2, 2))
        )

        conv5 = pipe(
            conv4_out,
            self.conv_block(1024, 3),
            Dropout(0.25)
        )

        up_side = pipe(
            conv5,
            self.conv_up_block(512, conv4),
            self.conv_up_block(256, conv3),
            self.conv_up_block(128, conv2),
            self.conv_up_block(64, conv1)
        )

        output = pipe(
            up_side,
            Conv2D(2, 3, activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal'),
            Conv2D(1, 1, activation = 'sigmoid')
        )

        with tf.device("/cpu:0"):
            model = Model(inputs=inputs, outputs=output)

        if self.weights_path:
            print(f'Loading weights from {self.weights_path}')
            model.load_weights(self.weights_path)

        if self.gpu_count > 1:
            model = multi_gpu_model(model, gpus=self.gpu_count, cpu_merge=False)

        return model

    def tune_tolerance(self):
        """
        Tune tolerance (1 - threshold) for setting as the default API parameter

        (Uses GPU to reduce computation from ~ 1 hour to a matter of seconds)
        """

        if len(self.val_generator) > 1:
            warnings.warn('Less than the entire validation batch will be used...')

        images, truth_masks = self.val_generator.__getitem__(0)
        print(f'Tuning of the tolerance parameter will occur on {images.shape[0]} images.')

        predictions = self.model.predict(images)

        tolerances = np.linspace(0.02, 1, 50)
        results = self._calculate_f1_scores(tolerances, truth_masks, predictions)

        self._plot_tolerances(tolerances, results)

        populations = [(f'{tolerance:.2f}', np.median(scores), np.std(scores))
                         for tolerance, scores in zip(tolerances, results)]

        tolerance, f1_median, f1_stdev = min(topk(5, populations, key=second), key=third)

        print(f'Tuned tolerance: {tolerance} w/ median={f1_median:.4f} stdev={f1_stdev:.4f}')
        return float(tolerance)

    def _calculate_f1_scores(self, tolerances, truth_masks, predictions):
        combined = np.array(list(zip(truth_masks, predictions)))

        tf.InteractiveSession()

        print('Calculating F1-Scores... This may take perhaps even an hour if no GPU.')
        start = time()
        results = abfs.keras.metrics.f1_scores_per_tolerances_tf(
            tf.constant(tolerances, tf.float32),
            tf.stack(combined)
        ).eval()
        print(f'F1-Score calculation complete: {time() - start:.2f} seconds')

        return results

    def _plot_tolerances(self, tolerances, results):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        file_path = tempfile.NamedTemporaryFile('w+b', suffix='.png', delete=False)

        def format_ticks(x, pos):
            return f'{x:.2f}' if pos % 3 == 0 else ''

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.boxplot(list(results), positions=tolerances, widths=0.015, sym='.')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.set_xlabel('Tolerance')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score by Adjusting Tolerance')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))

        plt.savefig(file_path.name)
        print(f'Plot has been saved to {file_path.name}. Please open to view.')

    def _callbacks(self):
        model_format = 'unet-%s-{epoch:04d}-{val_loss:.2f}.hdf5' % self.uuid
        log_dir = 'logs/unet-%s' % self.uuid

        return [
            TerminateOnNaN(),
            CustomTensorBoard(log_dir, self.data, self.shape),
            MultiGPUCheckpoint('checkpoints/' + model_format,
                            save_best_only=True, period=1)
        ]

class MultiGPUCheckpoint(ModelCheckpoint):
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model
