import numpy as np
import tensorflow as tf
from abfs.keras.metrics import mean_iou
from pytest import approx

def test_mean_iou():
    predicted = np.expand_dims(_predicted_100x100(), axis=0)
    actual = np.expand_dims(_actual_100x100(), axis=0)

    sess = tf.InteractiveSession()

    pred_t = tf.constant(predicted)
    actual_t = tf.constant(actual)

    tensor = mean_iou((100, 100), actual_t, pred_t)
    tf.global_variables_initializer().run()

    result = tensor.eval()

    expected_iou = 400 / (2 * 1600 - 400)

    assert result == approx(expected_iou, abs=0.0000001)

    sess.close()

def _predicted_100x100():
    matrix = np.zeros((100, 100), dtype=np.float32)
    matrix[20:60, 20:60] = 1
    return matrix

def _actual_100x100():
    matrix = np.zeros((100, 100), dtype=np.float32)
    matrix[40:80, 40:80] = 1
    return matrix
