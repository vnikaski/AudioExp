import numpy as np
import warnings
import pandas as pd
import os
import librosa
from tqdm import tqdm

from preprocessing.spectrograms import get_cochleagram


def CV_to_cochlea(dataset_dir, save_dir, sr=16000, offset=0):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    df = pd.read_table(os.path.join(dataset_dir,os.path.join('en', 'validated.tsv')))
    df = df[['path', 'sentence', 'age', 'gender']].dropna()

    files_loc = os.path.join(dataset_dir,os.path.join('en', 'clips'))

    for index, row in (pbar:= tqdm(df.iterrows())):
        pbar.set_description(f'{row["path"]}')
        f = os.path.join(files_loc, row["path"])
        if os.path.isfile(f):
            wav, _ = librosa.load(f, sr=sr, duration=2, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec = get_cochleagram(wav, sr)
            np.save(os.path.join(save_dir, row["path"]+'.npy'), spec)


CV_to_cochlea('/Volumes/Folder1/cv-corpus-12.0-delta-2022-12-07', '/Volumes/Folder1/cv-corpus-12.0-delta-2022-12-07/en/cochleagrams/')
