import numpy as np
from tensorflow import keras
import pandas as pd
import os


class FSD50K_preprocessed(keras.utils.Sequence):
    def __init__(self, data_path, classes=None, batch_size=32, shuffle=True, test_mode=False, val_mode=False, sr=22050, window_s = 1, n_mels=1024, n_fft=2048, hop=44):
        print("initialising generator...")
        self.data_path = data_path
        self.classes = classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.test_mode = test_mode
        if test_mode:
            gt = pd.read_csv(data_path+'FSD50K.ground_truth/eval.csv')
        else:
            gt = pd.read_csv(data_path+'FSD50K.ground_truth/dev.csv')
            if val_mode:
                gt = gt.loc[gt['split']=='val']
            else:
                gt = gt.loc[gt['split']=='train']

        gt['labels'] = gt['labels'].str.split(',')
        self.gt = gt[['fname', 'labels']]

        if self.classes is None:
            self.classes = set()
            for _, row in self.gt.iterrows():
                self.classes.update(set(row['labels']))
            self.n_classes = len(self.classes)

        else:
            self.n_classes = len(classes)
            self.gt['labels'] = self.gt['labels'].map(lambda x: [y for y in x if y in classes])
            self.gt = self.gt.loc[self.gt['labels'].str.join('') != '']  # deleting instances with no classes left
        self.sr = sr
        self.window_s = window_s
        self.hop = hop
        self.n_mels = n_mels
        self.n_fft= n_fft
        self.gt = self.gt.reset_index(drop=True)
        self.available_ids = list(self.gt.index)
        self.on_epoch_end()
        print('generator fully initialised')

    def __len__(self):
        return int(np.floor(len(self.gt) / self.batch_size))

    def __getitem__(self, id): # returns one batch
        batch_ids = self.indices[id*self.batch_size : (id+1)*self.batch_size]
        X, working = self._data_generation(batch_ids)
        y = self.gt.iloc[working]['labels']
        return X, self._categorical_labels(y)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.available_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _categorical_labels(self, y):
        return np.asarray([[1 if cls in y[i] else 0 for cls in self.classes] for i in y.index])

    def _data_generation(self, batch_ids):
        width = int((self.sr * self.window_s)/self.hop)
        X = np.empty((self.batch_size, self.n_mels, width, 1))
        batch_diff = 0
        working = []
        if self.test_mode:
            data_loc = 'eval_spectrograms'
        else:
            data_loc = 'dev_spectrograms'

        for i, id in enumerate(batch_ids):
            fname = self.gt.iloc[id]['fname']
            try:
                spec = np.load(os.path.join(self.data_path, f'{data_loc}/M{self.n_mels}_F{self.n_fft}/{fname}.wav.npy'))
            except:
                print(f'loading file {fname} raised an exception, this file was skipped')
                batch_diff += 1
                new_X = np.empty((self.batch_size-batch_diff, self.n_mels, width, 1))
                new_X[:i-batch_diff+1] = X[:i-batch_diff+1]
                X = new_X
                continue
            working.append(id)
            if spec.shape[0] != self.n_mels:
                raise ValueError(f"Got spectrogram with {spec.shape[0]} mels, {self.n_mels} expected")
            if spec.shape[1] > width:
                begin = np.random.randint(0, spec.shape[1]-width)
                spec = spec[:, begin : begin + width]
                X[i-batch_diff,] = spec.reshape((*spec.shape,1))
            else:
                pad_left = (width - spec.shape[1]) // 2
                pad_right = width - spec.shape[1] - pad_left
                spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=-80)
                X[i-batch_diff] = spec.reshape((*spec.shape,1))
        return X, working


