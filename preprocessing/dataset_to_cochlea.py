import numpy as np

from preprocessing.spectrograms import get_cochleagram
import pandas as pd
import os
import librosa
from tqdm import tqdm


def CV_to_cochlea(dataset_dir, save_dir, sr=16000):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    files_loc = os.path.join(dataset_dir,os.path.join('en', 'clips'))
    for filename in (pbar:= tqdm(os.listdir(files_loc))):
        pbar.set_description(f'{filename}')
        f = os.path.join(files_loc, filename)
        if os.path.isfile(f):
            wav, _ = librosa.load(f, sr=sr)
            spec = get_cochleagram(wav, sr)
            np.save(os.path.join(save_dir, filename+'.npy'), spec)
