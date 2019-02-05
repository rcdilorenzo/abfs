import tensorflow as tf
from toolz.curried import curry
import keras.backend as K

@curry
def mean_iou(image_shape, actual, pred):
    @tf.contrib.eager.defun
    def _mean_iou(tensor):
        pred_bool = tf.cast(pred, tf.bool)
        actual_bool = tf.cast(actual, tf.bool)

        intersection_mask = tf.logical_and(pred_bool, actual_bool)
        union_mask = tf.logical_or(pred_bool, actual_bool)

        intersection = tf.reduce_sum(tf.cast(intersection_mask, tf.float32), name='intersection')
        union = tf.reduce_sum(tf.cast(union_mask, tf.float32), name='union')

        return intersection / union

    iou = tf.map_fn(_mean_iou, (actual, pred), dtype=tf.float32)

    return K.mean(iou)

