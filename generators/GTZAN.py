import librosa
import numpy as np
from tensorflow import keras
import pandas as pd
import os
from preprocessing.spectrograms import get_mel_spectrogram
import stopit


class GTZAN(keras.utils.Sequence):
    def __init__(self, data_path, mode='train', batch_size=32, shuffle=True, window_s=1, sr=22050, n_mels=512, n_fft=2048, hop=44, quiet=False):
        print('initialising GTZAN generator...')
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        if sr != 22050:
            self.resample = True
        else:
            self.resample = False

        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop = hop
        self.window_s = window_s
        self.quiet = quiet
        self.mode = mode
        self.index = pd.read_csv(os.path.join(self.data_path, 'gtzan.index'))
        self.index = self.index.loc[self.index['split']==self.mode]
        self.data_path = os.path.join(self.data_path, 'genres_original')

        self.classes = np.asarray(self.index['label'].unique())

        self.available_ids = list(self.index.index)
        self.on_epoch_end()
        print(f'Generator fully initialised, {len(self.index)} samples available')

    def __len__(self):
        return int(np.floor(len(self.index) / self.batch_size))

    def __getitem__(self, id):
        batch_ids = self.indices[id*self.batch_size: (id+1)*self.batch_size]
        X, y = self._data_generation(batch_ids)
        return X, y

    def get_sample(self):
        X, y = self._data_generation([0])
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.available_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_generation(self, batch_ids):
        width = int((self.sr * self.window_s)/self.hop)
        X = np.empty((len(batch_ids), self.n_mels, width, 1))
        y = np.empty((len(batch_ids), len(self.classes)))
        batch_diff = 0

        for i, id in enumerate(batch_ids):
            codename = self.index.iloc[id]['fname'].split('.')
            wavname = f'{codename[0]}.{codename[1]}.{codename[-1]}'
            k = int(codename[2])
            fname = os.path.join(self.data_path, os.path.join(self.index.iloc[id]['label'], wavname))
            try:
                with stopit.ThreadingTimeout(2) as context_manager:
                    wavf, _ = librosa.load(fname, sr=22050)
                if context_manager.state == context_manager.TIMED_OUT:
                    raise stopit.TimeoutException
            except:
                if not self.quiet:
                    print(f'loading file {fname} raised an exception, this file was skipped')
                batch_diff += 1
                new_X = np.empty((len(batch_ids)-batch_diff, self.n_mels, width, 1))
                new_y = np.empty((len(batch_ids)-batch_diff, len(self.classes)))
                new_X[:i-batch_diff+1] = X[:i-batch_diff+1]
                new_y[:i-batch_diff+1] = y[:i-batch_diff+1]
                X = new_X
                y = new_y
                continue
            if self.resample:
                wavf = librosa.resample(wavf, orig_sr=22050, target_sr=self.sr)
            if len(wavf) >= int(self.sr*self.window_s):
                begin = np.random.randint(k*3*self.sr, (k*3+2)*self.sr)
                wavf = wavf[begin : begin+int(self.sr*self.window_s)]
            else:
                wavf = librosa.util.pad_center(wavf, size=int(self.sr*self.window_s))

            spec = get_mel_spectrogram(wavf, self.sr, self.n_fft, self.hop, self.n_mels)
            X[i-batch_diff] = spec[:,:X.shape[2]].reshape(X.shape[1:])
            label = self.index.iloc[id]['label']
            y[i-batch_diff] = (label == self.classes).astype('int')
        return X, y
