import tensorflow as tf
import keras.backend as K

def mean_iou(y_true, y_pred):
   score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score
