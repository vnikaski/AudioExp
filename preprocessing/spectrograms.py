import librosa
import numpy as np
from pycochleagram import cochleagram as cgram
from PIL import Image

n=50
low_lim, hi_lim = 20, 8000
nonlinearity, fft_mode, ret_mode = 'power', 'auto', 'envs'
sample_factor, pad_factor, downsample = 4, 2, 200
strict = True


def get_mel_spectrogram(signal, sr, n_fft, hop_length, n_mels):
    mspect = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mspect_db = librosa.power_to_db(mspect, ref=np.max)
    return mspect_db


def get_spectrogram(signal, n_fft, hop_length):
    spect = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    spect_db = librosa.amplitude_to_db(spect, ref=np.max)
    return spect_db


def resample(example, new_size):
    im = Image.fromarray(example)
    resized_image = im.resize(new_size, resample=Image.ANTIALIAS)
    return np.array(resized_image)


def get_cochleagram(signal, sr, window_s=2):
    c_gram = cgram.cochleagram(signal, sr, n, low_lim, hi_lim, sample_factor, pad_factor, downsample,
                      nonlinearity, fft_mode, ret_mode, strict)
    c_gram_rescaled =  255*(1-((np.max(c_gram)-c_gram)/np.ptp(c_gram)))
    c_gram_reshape_1 = np.reshape(c_gram_rescaled, (211,int(200*window_s)))
    c_gram_reshape_2 = resample(c_gram_reshape_1,(256,256))
    return c_gram_reshape_2
