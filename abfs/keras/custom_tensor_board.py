import io
import numpy as np
import tensorflow as tf
from PIL import Image

from keras.callbacks import TensorBoard

class CustomTensorBoard(TensorBoard):
    def __init__(self, log_dir, data, shape):
        self.data = data
        self.shape = shape
        super().__init__(log_dir=log_dir)

    def _predict(self, image):
        return self.model.predict(np.expand_dims(image, axis=0))[0]

    def _tf_image_value(self, image):
        height, width, channel = image.shape

        # Convert to PNG string-encoded data format
        output = io.BytesIO()
        Image.fromarray(image).save(output, format='PNG')
        image_data = output.getvalue()
        output.close()

        # Generate Tensorflow summary objects
        tf_image = tf.Summary.Image(
            height=height, width=width,
            colorspace=channel,
            encoded_image_string=image_data)

        return tf.Summary.Value(image=tf_image)

    def on_epoch_end(self, epoch, logs=None):
        batch_ids = [0, 3, 4, 5]
        values = list([self._tf_image_value(
            self.data
            .val_batch_data(i)
            .sample_image_predict(self.shape, self._predict)
            .astype(np.uint8)
        ) for i in batch_ids])

        tf_summary = tf.Summary(value=values)

        self.writer.add_summary(tf_summary, epoch)

        super().on_epoch_end(epoch, logs)
