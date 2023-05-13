from generators.Libritts import LibriTTSClean
from generators.GTZAN import GTZAN
from activation.maximiser import Maximiser
from models.models import Kell2018


import numpy as np
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--datapath')
parser.add_argument('-l', '--layername')
parser.add_argument('-s', '--savepath')
parser.add_argument('-b', '--batchsize')

args = parser.parse_args()

libri_gen = LibriTTSClean(
    data_path=args.datapath,
    words=1000,
    batch_size=int(args.batchsize),
    window_s=2,
    which_word=1,
    sr=16000,
    augment=False,
    norm='none',
    extend=False,
    preprocess='cochlea'
)

model = Kell2018(input_shape=(256,256,1), wout_shape = 589, gout_shape = 43, pretrained=True)
wkey = np.load('models/logits_to_word_key.npy')
gkey = np.load('models/logits_to_genre_key.npy')
w_weights = np.load('models/network_weights_word_branch.npy', allow_pickle=True, encoding='latin1')
g_weights = np.load('models/network_weights_genre_branch.npy', allow_pickle=True, encoding='latin1')
w_weights = w_weights.item()
g_weights = g_weights.item()
for lname in list(w_weights.keys()):
    model.get_layer(lname).set_weights([
        w_weights[lname]['W'],
        w_weights[lname]['b']
    ])
for lname in list(g_weights.keys()):
    model.get_layer(lname).set_weights([
        g_weights[lname]['W'],
        g_weights[lname]['b']
    ])

maxim = Maximiser()
maxim.init_submodel(model=model, out_layer_name=args.layername)

import warnings
warnings.filterwarnings("ignore")

df_most, df_least = maxim.look_for_all_channels(libri_gen)

df_most.to_csv(os.path.join(args.savepath, f"{args.layername}_most_libri.csv"))
df_least.to_csv(os.path.join(args.savepath, f"{args.layername}_least_libri.csv"))
