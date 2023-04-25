import numpy as np

from spectrograms import get_mel_spectrogram
import os
import librosa

# N_MELS = 1024  # should correspond to similiar resolution to human ear
N_MELS = 256
N_FFT = 2048 # around 100 ms for sr=22050

def dataset_to_mel(dataset_dir, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(dataset_dir, f'M{N_MELS}_F{N_FFT}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    i = 0
    for filename in os.listdir(dataset_dir):
        print(f'{filename} {i}')
        if i>=29874:
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


