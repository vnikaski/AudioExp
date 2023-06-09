import numpy as np
from tensorflow import keras
import pandas as pd
import os
import tensorflow as tf
from generators.Libritts import LibriTTSClean
from generators.GTZAN import GTZAN

class TTSGenre(keras.utils.Sequence):
    def __init__(self, libri_path, gtzan_path, mode='train', batch_size=64, shuffle=True, window_s=1, sr=22050, n_mels=512, n_fft=2048, hop=44, words=200, which_word=2, quiet=False, augment=True, norm='sample', urbanpath=None, wbatch=None, gbatch=None):
        if wbatch is None and gbatch is None:
            wbatch = batch_size//2
            gbatch = batch_size//2
        elif gbatch is None:
            wbatch = int(wbatch)
            gbatch = batch_size-wbatch
        elif wbatch is None:
            gbatch = int(gbatch)
            wbatch = batch_size-gbatch
        else:
            gbatch = int(gbatch)
            wbatch = int(wbatch)
        self.mode = mode

        self.libriGen = LibriTTSClean(data_path=libri_path,
                                      mode=mode,
                                      words=words,
                                      batch_size=wbatch,
                                      shuffle=shuffle,
                                      window_s=window_s,
                                      which_word=which_word,
                                      sr=sr,
                                      n_mels=n_mels,
                                      n_fft=n_fft,
                                      hop=hop,
                                      quiet=quiet,
                                      norm=norm,
                                      augment=augment,
                                      urban_path=urbanpath)
        self.gtzanGen = GTZAN(data_path=gtzan_path,
                              mode=mode,
                              batch_size=gbatch,
                              shuffle=shuffle,
                              window_s=window_s,
                              sr=sr,
                              n_mels=n_mels,
                              n_fft=n_fft,
                              hop=hop,
                              quiet=quiet,
                              norm=norm)
        self.classes={'wout': np.zeros((0,len(self.libriGen.words))), 'gout':np.zeros((0,10))}

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
        if self.mode in ['val', 'test']:
            self.classes['wout'] = np.concatenate([self.classes['wout'], wout])
            self.classes['gout'] = np.concatenate([self.classes['gout'], gout])
        wout = tf.constant(wout)
        gout = tf.constant(gout)
        return keras.backend.variable(X), {'wout': wout, 'gout': gout}

    def get_words(self):
        return self.libriGen.words

    def on_epoch_end(self):
        self.libriGen.on_epoch_end()
        self.gtzanGen.on_epoch_end()

    def get_sample(self):
        X_l, y_l = self.libriGen.get_sample()
        X_g, y_g = self.gtzanGen.get_sample()
        X = np.concatenate([X_l, X_g])
        wout = np.zeros((2, *y_l.shape[1:]))
        gout = np.zeros((2, *y_g.shape[1:]))
        wout[0] = y_l
        gout[1] = y_g
        return X, {'wout': wout, 'gout': gout}
