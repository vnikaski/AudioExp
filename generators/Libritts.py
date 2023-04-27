import librosa
import numpy as np
from tensorflow import keras
import pandas as pd
import os
from preprocessing.spectrograms import get_mel_spectrogram
import stopit

class LibriTTSClean(keras.utils.Sequence):
    def __init__(self, data_path, mode='train', words=200, batch_size=32, shuffle=True, window_s=1, which_word=2, sr=24000, n_mels=512, n_fft=2048, hop=44, quiet=False):
        print('initialising Libri generator...')
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        if sr != 24000:
            self.resample=True
        else:
            self.resample=False
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop = hop
        self.window_s = window_s
        self.quiet = quiet
        self.mode = mode
        if self.mode == 'train':
            self.subset = 'train-clean-360'
        elif self.mode == 'val':
            self.subset = 'dev-clean'
        elif self.mode == 'test':
            self.subset = 'test-clean'
        else:
            raise ValueError(f'Unsupported mode {self.mode}, try \'train\', \'val\' or \'test\'')
        self.data_path = os.path.join(self.data_path, self.subset+'/')

        if not os.path.exists(self.data_path+self.subset+'.index'):
            index_df = pd.DataFrame(columns=['fname', 'word1', 'word2', 'word3'])
            for speaker in os.listdir(self.data_path):
                if os.path.isdir(os.path.join(self.data_path, speaker)):
                    for chapter in os.listdir(os.path.join(self.data_path, speaker)):
                        if os.path.isdir(os.path.join(os.path.join(self.data_path, speaker),chapter)):
                            try:
                                ch_trans = pd.read_table(f'{os.path.join(os.path.join(self.data_path, speaker),chapter)}/{speaker}_{chapter}.trans.tsv', header=None)
                            except pd.errors.ParserError:
                                print(f'speaker {speaker} with chapter {chapter} was ommitted due to parsing issues')
                                continue
                            except OSError as e:
                                print(f'speaker {speaker} with chapter {chapter} was ommitted due to {e}')
                                continue

                            ch_trans['words'] = ch_trans[1].str.lower().str.replace('[^a-zA-Z ]', '').str.split()
                            for loc, row in ch_trans.iterrows():
                                wlen = len(row['words'])
                                index_row = {'fname': row[0], 'word1': pd.NA, 'word2': pd.NA, 'word3': pd.NA}
                                if wlen>=1:
                                    index_row['word1'] = row['words'][0]
                                if wlen>=2:
                                    index_row['word2'] = row['words'][1]
                                if wlen>=3:
                                    index_row['word3'] = row['words'][2]
                                index_df = pd.concat([index_df,pd.DataFrame([index_row])], ignore_index=True)
                                index_df.to_csv(f'{os.path.join(os.path.join(self.data_path, speaker),chapter)}{self.subset}.index')
                                self.index = index_df

        else:
            self.index = pd.read_csv(self.data_path+self.subset+'.index')

        if which_word not in [1,2,3]:
            raise ValueError(f'which_word should be 1, 2 or 3; got {which_word}')
        self.index['word'] = self.index[f'word{which_word}']
        self.index = self.index[['fname', 'word']]

        if type(words) is int:
            words = list(self.index['word'].value_counts()[:words].index)

        self.words = np.asarray(words)
        self.index = self.index[self.index['word'].isin(words)].reset_index(drop=True)

        self.available_ids = list(self.index.index)
        self.on_epoch_end()
        print(f'Generator fully initialised, {len(self.index)} samples available')

    def __len__(self):
        return int(np.floor(len(self.index) / self.batch_size))

    def __getitem__(self, id): # returns one batch
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
        y = np.empty((len(batch_ids), len(self.words)))
        batch_diff = 0

        for i,id in enumerate(batch_ids):
            loc = self.index.iloc[id]['fname'].split('_')
            fname = os.path.join(os.path.join(self.data_path, f'{loc[0]}/{loc[1]}'), self.index.iloc[id]['fname']) + '.wav'
            try:
                with stopit.ThreadingTimeout(10) as context_manager:
                    wavf, _ = librosa.load(fname, sr=24000)
                if context_manager.state == context_manager.TIMED_OUT:
                    raise stopit.TimeoutException
            except Exception as e:
                if not self.quiet:
                    print(f'loading file {fname} raised an exception {e}, this file was skipped')
                batch_diff += 1
                new_X = np.empty((len(batch_ids)-batch_diff, self.n_mels, width, 1))
                new_y = np.empty((len(batch_ids)-batch_diff, len(self.words)))
                new_X[:i-batch_diff+1] = X[:i-batch_diff+1]
                new_y[:i-batch_diff+1] = y[:i-batch_diff+1]
                X = new_X
                y = new_y
                continue
            if self.resample:
                wavf = librosa.resample(wavf, orig_sr=24000, target_sr=self.sr)

            if len(wavf) >= int(self.sr*self.window_s):
                wavf = wavf[:int(self.sr*self.window_s)]
            else:
                wavf = librosa.util.pad_center(wavf, size=int(self.sr*self.window_s))

            spec = get_mel_spectrogram(wavf, self.sr, self.n_fft, self.hop, self.n_mels)

            X[i-batch_diff] = spec[:,:X.shape[2]].reshape(X.shape[1:])
            word = self.index.iloc[id]['word']
            y[i-batch_diff] = (word == self.words).astype('int')
        return X, y



