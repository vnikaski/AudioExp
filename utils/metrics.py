import keras.backend
import tensorflow as tf
from keras.metrics import Accuracy



class UnconcernedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='unconcerned_accuracy', **kwargs):
        super(UnconcernedAccuracy, self).__init__(name=name, **kwargs)
        self.unconcerned_acc = self.add_weight(name='uacc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        indices = tf.reduce_sum(y_true, axis=1) != 0
        print(indices)

        if sample_weight is not None:
            sample_weight = sample_weight[indices]

        acc = Accuracy()
        acc.update_state(y_true[indices], y_pred[indices], sample_weight)
        self.unconcerned_acc.assign_add(acc.result())

    def result(self):
        return self.unconcerned_acc
