from activation.maximiser import Maximiser
from models.models import Kell2018

import os
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import argparse

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

input_shape = (256,256,1)
target_shape = 256*256

parser = argparse.ArgumentParser()

parser.add_argument('--datapath')
parser.add_argument('--mode', choices=['gender', 'gender2', 'age'])
parser.add_argument('--savepath')
args = parser.parse_args()

datapath = args.datapath

validated = pd.read_table(os.path.join(datapath, 'validated.tsv'))
validated = validated[['path', 'sentence', 'age', 'gender']]
df = validated.dropna().reset_index(drop=True)
df = df.drop([2437]).reset_index(drop=True)
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

datapath = os.path.join(datapath, 'cochleagrams')

print('loading data ...')

train_X = np.empty((len(df), *input_shape))
for index, fname in enumerate(df['path']):
    train_X[index] = np.load(os.path.join(datapath, fname+'.npy')).reshape(input_shape)

print('... data loaded :)')


cut_indices = np.random.choice(list(range(len(df))), int(len(df)*0.6), replace=False)

df = df.drop(cut_indices).reset_index(drop=True)
train_X = np.delete(train_X, cut_indices, 0)

if args.mode == 'gender':
    y = df['gender']
    classes = list(df['gender'].unique())
elif args.mode == 'age':
    y = df['age']
    classes = list(df['age'].unique())
else:
    raise NotImplementedError

assert len(np.where(pd.isna(train_X))[0]) == 0

print('base')

layer = 'base'

clf = LogisticRegression(max_iter=1000)

"""
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

for i, (train_index, test_index) in enumerate(skf.split(X=train_X, y=y)):
    clf = SGDClassifier(loss='log')
"""

scores = cross_val_score(clf, train_X.reshape((train_X.shape[0], target_shape)), y, cv=10, verbose=3)
np.save(os.path.join(args.savepath, f'{args.mode}_{layer}.npy'), scores)

print('loading model...')

model = Kell2018(input_shape=(256,256,1), wout_shape = 589, gout_shape = 43, pretrained=True)
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

print('...model loaded :)')


layers = ['relu1', 'max_pool1', 'relu2', 'max_pool2', 'relu3', 'relu4_W', 'relu5_W', 'avg_pool5_W', 'fc6_W', 'fctop_W', 'relu4_G', 'relu5_G', 'avg_pool5_G', 'fc6_G', 'fctop_G']

for layer in layers:
    print(layer)
    maxim = Maximiser()
    maxim.init_submodel(model=model, out_layer_name=layer)
    rep_X = maxim.submodel.predict(train_X)
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(
        clf,
        np.reshape(rep_X, newshape=(rep_X.shape[0], np.prod(rep_X.shape[1:]))),
        y,
        cv=10,
        verbose=3,
    )
    del rep_X
    np.save(os.path.join(args.savepath, f'{args.mode}_{layer}.npy'), scores)
