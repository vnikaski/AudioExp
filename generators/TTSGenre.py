import numpy as np
from tensorflow import keras
import pandas as pd
import os
import tensorflow as tf
from generators.Libritts import LibriTTSClean
from generators.GTZAN import GTZAN

class TTSGenre(keras.utils.Sequence):
    def __init__(self, libri_path, gtzan_path, mode='train', batch_size=64, shuffle=True, window_s=1, sr=22050, n_mels=512, n_fft=2048, hop=44, words=200, which_word=2):
        self.libriGen = LibriTTSClean(data_path=libri_path,
                                      mode=mode,
                                      words=words,
                                      batch_size=batch_size//2,
                                      shuffle=shuffle,
                                      window_s=window_s,
                                      which_word=which_word,
                                      sr=sr,
                                      n_mels=n_mels,
                                      n_fft=n_fft,
                                      hop=hop)
        self.gtzanGen = GTZAN(data_path=gtzan_path,
                              mode=mode,
                              batch_size=batch_size//2,
                              shuffle=shuffle,
                              window_s=window_s,
                              sr=sr,
                              n_mels=n_mels,
                              n_fft=n_fft,
                              hop=hop)
    def __len__(self):
        return min(len(self.gtzanGen), len(self.libriGen))

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i==self.__len__()-1:
                self.on_epoch_end()

    def __getitem__(self, id):
        X_l, y_l = self.libriGen[id]
        X_g, y_g = self.gtzanGen[id]
        X = np.concatenate([X_l, X_g])
        te = X.shape[0]
        assert y_l.shape[0] + y_g.shape[0] == te
        wout = np.zeros((te, *y_l.shape[1:]))
        gout = np.zeros((te, *y_g.shape[1:]))
        wout[:y_l.shape[0]] = y_l
        gout[y_l.shape[0]:] = y_g
        wout = tf.constant(wout)
        gout = tf.constant(gout)
        return keras.backend.variable(X), {'wout': wout, 'gout': gout}

    def get_sample(self):
        X_l, y_l = self.libriGen.get_sample()
        X_g, y_g = self.gtzanGen.get_sample()
        X = np.concatenate([X_l, X_g])
        wout = np.zeros((2, *y_l.shape[1:]))
        gout = np.zeros((2, *y_g.shape[1:]))
        wout[0] = y_l
        gout[1] = y_g
        return X, {'wout': wout, 'gout': gout}
