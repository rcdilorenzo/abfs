from funcy import rpartial
from toolz import memoize, pipe, curry

import h5py
import math
import uuid

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

class UNet():
    def __init__(self, data, shape,
                 max_batches=math.inf,
                 gpu_count=1,
                 epochs=100,
                 learning_rate=0.001):

        self.uuid = uuid.uuid4().hex[0:6]
        self.data = data
        self.shape = shape
        self.gpu_count = gpu_count
        self.max_batches = max_batches
        self.epochs = epochs
        self.learning_rate = learning_rate

    def mean_iou(self, actual, pred):
        return abfs.keras.metrics.mean_iou(actual, pred)

    def train(self):
        model = self.model

        if self.gpu_count > 1:
            model = multi_gpu_model(model, gpus=self.gpu_count, cpu_merge=False)

        model.compile(optimizer=SGD(lr=self.learning_rate),
                           loss='binary_crossentropy',
                           metrics=[self.mean_iou])

        model.fit_generator(self.train_generator,
                            validation_data=self.val_generator,
                            epochs=self.epochs,
                            callbacks=self._callbacks())

    @property
    @memoize
    def train_generator(self):
        return self.data.train_generator(Generator,
                                         self.shape,
                                         max_batches=self.max_batches)

    @property
    @memoize
    def val_generator(self):
        return self.data.val_generator(Generator,
                                       self.shape,
                                       max_batches=self.max_batches)

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
            return Model(inputs=inputs, outputs=output)

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
