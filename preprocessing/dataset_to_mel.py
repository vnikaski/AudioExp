import numpy as np

from preprocessing.spectrograms import get_mel_spectrogram
import pandas as pd
import os
import librosa

# N_MELS = 1024  # should correspond to similiar resolution to human ear
N_MELS = 256
N_FFT = 2048 # around 100 ms for sr=22050
WINDOW_S = 1

def FSD50K_to_mel(dataset_dir, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(dataset_dir, f'M{N_MELS}_F{N_FFT}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    i = 0
    for filename in os.listdir(dataset_dir):
        print(f'{filename} {i}')
        if i>=29874: # hardcoded where the conversion stopped last time
            f = os.path.join(dataset_dir, filename)
            if os.path.isfile(f):
                wav, sr = librosa.load(f)
                wav, _ = librosa.effects.trim(wav)
                if sr != 22050:
                    raise Warning(f'Sr different than 22050 Hz ({sr})')
                hop_length = int(0.002 * sr) # temporal resolution of ear: 2ms at best https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5063729/
                m_spec = get_mel_spectrogram(wav, sr, N_FFT, hop_length, N_MELS)
                np.save(os.path.join(save_dir, filename+'.npy'), m_spec)
        i += 1


def LibriTTS_to_mel(data_dir, save_dir, subset, words=200, which_word=2, resample=22050):
    sr = 24000 # sr of files in libritts
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data_dir = os.path.join(data_dir, subset)
    df = pd.read_csv(os.path.join(data_dir, f'{subset}.index'))
    df['word'] = df[f'word{which_word}']
    df = df[['fname', 'word']]

    if type(words) is int:
        words = list(df['word'].value_counts()[:words].index)

    words = np.asarray(words)
    df = df[df['word'].isin(words)].reset_index(drop=True)

    for fname in df['fname']:
        print(fname)
        loc = fname.split('_')
        filepath = os.path.join(os.path.join(data_dir, f'{loc[0]}/{loc[1]}'), fname) + '.wav'
        savepath = os.path.join(os.path.join(save_dir, f'{loc[0]}/{loc[1]}'), fname) + '.npy'

        if not os.path.exists(savepath): # so that the folder can be successively updated with more words
            if os.path.isfile(filepath):
                wav, _ = librosa.load(filepath,sr=sr)
            else:
                continue
            if resample:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=resample)
                sr = resample
            hop_length = int(0.002 * sr) # temporal resolution of ear: 2ms at best https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5063729/
            if len(wav) >= int(sr*WINDOW_S):
                wav = wav[:int(sr*WINDOW_S)]
            else:
                wav = librosa.util.pad_center(wav, size=int(sr*WINDOW_S))

            spec = get_mel_spectrogram(wav, sr, N_FFT, hop_length, N_MELS)
            if not os.path.exists(os.path.join(save_dir, f'{loc[0]}')):
                os.mkdir(os.path.join(save_dir, f'{loc[0]}'))
            if not os.path.exists(os.path.join(save_dir, f'{loc[0]}/{loc[1]}')):
                os.mkdir(os.path.join(save_dir, f'{loc[0]}/{loc[1]}'))
            np.save(savepath, spec)


def GTZAN_to_mel(data_dir, save_dir):
    pass



