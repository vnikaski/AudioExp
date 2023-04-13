import librosa
import numpy as np


def get_mel_spectrogram(signal, sr, n_fft, hop_length, n_mels):
    mspect = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mspect_db = librosa.power_to_db(mspect, ref=np.max)
    return mspect_db


def get_spectrogram(signal, n_fft, hop_length):
    spect = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    spect_db = librosa.amplitude_to_db(spect, ref=np.max)
    return spect_db
