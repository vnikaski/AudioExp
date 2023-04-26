import keras.backend
import tensorflow as tf
from keras.losses import CategoricalCrossentropy

cce = CategoricalCrossentropy()

class Unconcerned_CCE(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        indices = tf.reduce_sum(y_true, axis=1) != 0
        return cce(y_true[indices], y_pred[indices])

"""
def unconcerned_categorical_crossentropy(y_true, y_pred):
    print('computing')
    indices = tf.reduce_sum(y_true, axis=1) != 0
    loss = cce(
            y_true[indices],
            y_pred[indices]
        )
    return loss
"""
