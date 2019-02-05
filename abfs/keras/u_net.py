from funcy import rpartial
from toolz import memoize

import h5py
import math
import uuid

import abfs.keras.metrics
from abfs.keras.generator import Generator
from abfs.keras.custom_tensor_board import CustomTensorBoard

import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
import keras.backend as K

class UNet():
    def __init__(self, data, shape, max_batches=math.inf, epochs=100):
        self.uuid = uuid.uuid4().hex[0:6]
        self.data = data
        self.shape = shape
        self.max_batches = max_batches
        self.epochs = epochs

    def mean_iou(self, actual, pred):
        return abfs.keras.metrics.mean_iou(self.shape, actual, pred)

    def train(self):
        self.model.compile(optimizer=Adam(lr = 1e-4),
                           loss='binary_crossentropy',
                           metrics=[self.mean_iou])

        self.model.fit_generator(self.train_generator,
                                 validation_data=self.val_generator,
                                 epochs=self.epochs, callbacks=self._callbacks())

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


    @property
    @memoize
    def model(self):
        # Model based on https://github.com/zhixuhao/unet implementation of U-Net
        height, width = self.shape

        inputs = Input((height, width, 3))
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        return Model(input = inputs, output = conv10)

    def _callbacks(self):
        model_format = 'unet-%s-{epoch:04d}-{val_loss:.2f}.hdf5' % self.uuid
        log_dir = 'logs/unet-%s' % self.uuid

        return [
            TerminateOnNaN(),
            CustomTensorBoard(log_dir, self.data, self.shape),
            ModelCheckpoint('checkpoints/' + model_format,
                            save_best_only=True, period=1)
        ]
