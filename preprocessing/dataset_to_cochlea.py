import numpy as np
import warnings
import pandas as pd
import os
import librosa
from tqdm import tqdm

from preprocessing.spectrograms import get_cochleagram
from models.load_AST import load_AST


def CV_to_cochlea(dataset_dir, save_dir, sr=16000, offset=0):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    df = pd.read_csv(os.path.join(dataset_dir,'paths_df.csv'))

    files_loc = os.path.join(dataset_dir,'clips')

    for index, row in (pbar:= tqdm(df.iterrows(), total=len(df))):
        pbar.set_description(f'{row["path"]}')
        f = os.path.join(files_loc, row["path"])
        if os.path.isfile(f):
            wav, _ = librosa.load(f, sr=sr, duration=2, offset=offset)
            if len(wav) < sr*2:
                wav = librosa.util.pad_center(wav, size=sr*2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec = get_cochleagram(wav, sr)
            np.save(os.path.join(save_dir, row["path"]+'.npy'), spec)

def CV_to_mel(dataset_dir, save_dir, sr=16000, offset=0):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    df = pd.read_csv(os.path.join(dataset_dir, 'paths_df.csv'))

    files_loc = os.path.join(dataset_dir, 'clips')

    feature_extractor, model = load_AST()

    for index, row in (pbar:= tqdm(df.iterrows(), total=len(df))):
        pbar.set_description(f'{row["path"]}')
        f = os.path.join(files_loc, row["path"])
        if os.path.isfile(f):
            wav, _ = librosa.load(f, sr=sr, duration=2, offset=offset)
            if len(wav) < sr*2:
                wav = librosa.util.pad_center(wav, size=sr*2)
            spec = feature_extractor(wav, sampling_rate=sr, return_tensors="pt")['input_values'].numpy()
            np.save(os.path.join(save_dir, row["path"]+'.npy'), spec)


#CV_to_cochlea('/Users/vnika/Documents/studia/audio_exp/data/CommonVoice/', '/Users/vnika/Documents/studia/audio_exp/data/CommonVoice/cochleagrams/')

CV_to_mel('/Users/vnika/Documents/studia/audio_exp/data/CommonVoice/', '/Users/vnika/Documents/studia/audio_exp/data/CommonVoice/mel_spectrograms/')
