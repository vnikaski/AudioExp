import librosa
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import os
from preprocessing.spectrograms import get_mel_spectrogram, get_cochleagram
import stopit

NOISE_LVL = 0.5
GET_WAV = False # for testing purposes

class LibriTTSGender(keras.utils.Sequence):
    def __init__(self, data_path, mode='train', words=200, batch_size=32, shuffle=True, window_s=1, which_word=2, sr=24000, n_mels=512, n_fft=2048, hop=44, quiet=False, augment=True, norm='sample', urban_path=None, extend=True, preprocess='mel'):
        print(f'initialising {mode} Libri generator...')
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop = hop
        self.window_s = window_s
        self.quiet = quiet
        self.augment = augment
        self.norm = norm
        self.mode = mode
        if self.mode == 'train':
            self.subset = 'train-clean-360'
        elif self.mode == 'val':
            self.subset = 'dev-clean'
        elif self.mode == 'test':
            self.subset = 'test-clean'
        else:
            raise ValueError(f'Unsupported mode {self.mode}, try \'train\', \'val\' or \'test\'')

        self.genders = pd.read_table(os.path.join(self.data_path, 'speakers.tsv'), index_col=False, names=['reader', 'gender', 'subset', 'name'], header=0)
        self.gender_ans = np.asarray(['F', 'M'])

        self.data_path = os.path.join(self.data_path, self.subset+'/')
        self.preprocess = preprocess

        if self.augment:
            if urban_path is None:
                raise ValueError('please provide urban path with flag --urbanpath')
            self.urban_index = pd.read_csv(os.path.join(urban_path, 'urban.index'))
            self.urban_path = urban_path

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

        try:
            len(words)
        except:
            words = list(self.index['word'].value_counts()[:words].index)

        self.words = np.asarray(words)
        self.index = self.index[self.index['word'].isin(words)].reset_index(drop=True)

        print(f"maximal word presence before extension: {self.index.value_counts('word').values[0]}")
        print(f"minimal word presence before extension: {self.index.value_counts('word').values[-1]}")

        if extend and mode=='train': # todo: think how it should look, this is just a temporary fix
            target = self.index.value_counts('word').values[0]
            for word in words:
                focus = self.index[self.index['word']==word]
                for i in range((target//len(focus))-1):
                    self.index = pd.concat([self.index, focus])

            print(f"maximal word presence after extension: {self.index.value_counts('word').values[0]}")
            print(f"minimal word presence after extension: {self.index.value_counts('word').values[-1]}")

        self.available_ids = list(self.index.index)
        self.on_epoch_end()
        print(f'Generator fully initialised, {len(self.index)} samples available')

    def __len__(self):
        return int(np.floor(len(self.index) / self.batch_size))

    def __getitem__(self, id): # returns one batch
        batch_ids = self.indices[id*self.batch_size: (id+1)*self.batch_size]
        X, y = self._data_generation(batch_ids)
        return X, y

    def get_data_with_info(self, id):
        batch_ids = self.indices[id*self.batch_size: (id+1)*self.batch_size]
        X, _ = self._data_generation(batch_ids)
        df = self.index.iloc[batch_ids]
        return X, df

    def get_sample(self):
        X, y = self._data_generation([0])
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.available_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_generation(self, batch_ids):
        # todo: change int(self.sr*self.window_s) to width
        if self.preprocess == 'mel':
            width = int((self.sr * self.window_s)/self.hop)
            height = int(self.n_mels)
        elif self.preprocess == 'cochlea':
            width, height = 256, 256
        else:
            raise NotImplementedError
        X = np.empty((len(batch_ids), height, width, 1))
        y = np.empty((len(batch_ids), 2))
        batch_diff = 0

        for i,id in enumerate(batch_ids):
            loc = self.index.iloc[id]['fname'].split('_')
            gender = self.genders[self.genders['reader']==int(loc[0])]['gender'].values[0]
            fname = os.path.join(os.path.join(self.data_path, f'{loc[0]}/{loc[1]}'), self.index.iloc[id]['fname']) + '.wav'
            try:
                with stopit.ThreadingTimeout(10) as context_manager:
                    wavf, _ = librosa.load(fname, sr=self.sr)
                if context_manager.state == context_manager.TIMED_OUT:
                    raise stopit.TimeoutException
            except Exception as e:
                if not self.quiet:
                    print(f'loading file {fname} raised an exception {e}, this file was skipped')
                batch_diff += 1
                new_X = np.empty((len(batch_ids)-batch_diff, height, width, 1))
                new_y = np.empty((len(batch_ids)-batch_diff, 2))
                new_X[:i-batch_diff+1] = X[:i-batch_diff+1]
                new_y[:i-batch_diff+1] = y[:i-batch_diff+1]
                X = new_X
                y = new_y
                continue

            if len(wavf) >= int(self.sr*self.window_s):
                wavf = wavf[:int(self.sr*self.window_s)]
            else:
                wavf = librosa.util.pad_center(wavf, size=int(self.sr*self.window_s))

            if self.augment:
                which = np.random.randint(0, len(self.urban_index)-1)
                noise, _ = librosa.load(os.path.join(os.path.join(self.urban_path, 'audio'), self.urban_index['fname'][which]), sr=self.sr)
                if len(noise) >= int(self.sr*self.window_s):
                    begin = np.random.randint(0, len(noise)-int(self.sr*self.window_s)-1)
                    noise = noise[begin: begin+int(self.sr*self.window_s)]
                else:
                    noise = librosa.util.pad_center(noise, size=int(self.sr*self.window_s))
                noise *= NOISE_LVL
                wavf += noise

            if self.preprocess == 'mel':
                spec = get_mel_spectrogram(wavf, self.sr, self.n_fft, self.hop, self.n_mels)
            elif self.preprocess == 'cochlea':
                spec = get_cochleagram(wavf, self.sr, self.window_s)
            else:
                raise NotImplementedError
            if self.norm == 'sample':
                mean, var = tf.nn.moments(spec, axes=[0,1])
                spec = (spec-mean.numpy())/np.sqrt(var.numpy()+1e-8)  # prevent 0 division
            X[i-batch_diff] = spec[:,:X.shape[2]].reshape(X.shape[1:])
            y[i-batch_diff] = (gender == self.gender_ans).astype('int')
        if self.norm == 'batch':
            mean, var = tf.nn.moments(X, axes=[0,1,2,3])
            X = (X-mean.numpy())/tf.sqrt(var.numpy()+1e-8)
        return X, y
