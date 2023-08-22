from models.load_AST  import load_AST

import os
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import torch

from tqdm import tqdm

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.config.set_visible_devices([], 'GPU')

input_shape = (1024,128)
target_shape = 1024*128

parser = argparse.ArgumentParser()

parser.add_argument('--datapath')
parser.add_argument('--mode', choices=['gender', 'age'])
parser.add_argument('--savepath')
args = parser.parse_args()

datapath = args.datapath

validated = pd.read_csv(os.path.join(datapath, f'{args.mode}_df.csv'))
if args.mode == 'gender':
    validated = validated[['path', 'sentence', 'age', 'gender']]
else:
    validated = validated[['path', 'sentence', 'age_grouped', 'gender']]
    validated['age'] = validated['age_grouped']
df = validated.dropna().reset_index(drop=True)
# df = df.sample(frac=1, random_state=0).reset_index(drop=True)

datapath = os.path.join(datapath, 'mel_spectrograms')

print('loading data ...')

train_X = np.empty((len(df), *input_shape))
for index, fname in enumerate(df['path']):
    train_X[index] = np.load(os.path.join(datapath, fname+'.npy'))

print('... data loaded :)')


# cut_indices = np.random.choice(list(range(len(df))), int(len(df)*0.6), replace=False)

# df = df.drop(cut_indices).reset_index(drop=True)
# train_X = np.delete(train_X, cut_indices, 0)

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
np.save(os.path.join(args.savepath, f'{args.mode}_{layer}_AST.npy'), scores)

print('loading model...')

_, model = load_AST()

print('...model loaded :)')


layers = ['relu1', 'max_pool1', 'relu2', 'max_pool2', 'relu3', 'relu4_W', 'relu5_W', 'avg_pool5_W', 'fc6_W', 'fctop_W', 'relu4_G', 'relu5_G', 'avg_pool5_G', 'fc6_G', 'fctop_G']

N_HS = 13

device = torch.device('mps')

model.eval()
model.to(device)
for param in model.parameters():
    param.requires_grad = False

train_X = torch.Tensor(train_X).to(device)

loading_splits = 64

split_size = train_X.size(0)//loading_splits
assert loading_splits > 0
#print(len(train_X))
for k in tqdm(range(loading_splits)):
    if k == loading_splits-1:
        sub_rep = model(train_X[k*split_size:]).hidden_states
    else:
        sub_rep = model(train_X[k*split_size:(k+1)*split_size]).hidden_states

    for l in range(N_HS):
        np.save(f'sub_rep_{l}.npy',sub_rep[l].cpu().numpy())
    del sub_rep

    for l in range(N_HS):
        sub_rep = np.load(f'sub_rep_{l}.npy')
        if k==0:
            np.save(f'rep_{l}.npy', sub_rep)
        else:
            rep = np.load(f'rep_{l}.npy')
            rep = np.concatenate([rep, sub_rep])
            np.save(f'rep_{l}.npy', rep)

del sub_rep, rep

for i in range(N_HS):
    print(i)
    rep_X = np.load(f'rep_{i}.npy')
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(
        clf,
        np.reshape(rep_X, newshape=(rep_X.shape[0], np.prod(rep_X.shape[1:]))),
        y,
        cv=10,
        verbose=3,
    )
    np.save(os.path.join(args.savepath, f'{args.mode}_{i}_AST.npy'), scores)
    del rep_X
