from tqdm import tqdm
import pandas as pd
import numpy as np
import sklearn
import librosa
import librosa.display
import os
import argparse
import time
import tensorflow as tf
import keras
import keras.backend as K

from models.models import Kell2018
from preprocessing.spectrograms import get_cochleagram


def rescale_gradients(gradients):
    return gradients/(K.sqrt(K.mean(K.square(gradients), axis=[1,2], keepdims=True))+1e-6)


def lr_factor(step, warmup, total_steps):
    if step < warmup:
        return step*(1/warmup)
    elif step >= warmup:
        return 0.5 * (1 + np.cos(np.pi * (step-warmup) / (total_steps-warmup)))


def optimise_metamer(input_img, model, orig_activation, layer, n_steps, upward_lim=8, reduce_factor=0.5, prev_loss=None, save_dir=None, seed=None):
    if prev_loss is None:
        prev_loss=np.inf
    upward_count=0

    input_img = K.variable(input_img)
    prev_inp = tf.identity(input_img)
    lr = 0.001

    for j in (pbar:= tqdm(range(n_steps))):
        with tf.GradientTape() as gtape:
            outputs_t = model(input_img)
            hs = tf.math.add(outputs_t, -orig_activation)
            loss = tf.math.multiply(tf.norm(hs, axis=None, ord=2), 1/(tf.norm(orig_activation, ord=2)+1e-8))

        grads = gtape.gradient(loss, input_img)
        input_img.assign_sub(lr * lr_factor(step=j, warmup=256, total_steps=n_steps) * rescale_gradients(grads))

        if loss==0:
            return input_img, loss[0]

        if loss>prev_loss:
            if upward_count>=upward_lim:
                input_img = tf.identity(prev_inp)
                upward_count = 0
            else:
                upward_count += 1
        else:
            upward_count=0
            prev_loss = tf.identity(loss)
            prev_inp = tf.identity(input_img)

        input_img = K.variable(np.clip(input_img.numpy(), a_min=-1.5, a_max=1.5))

        if j%500 == 0 and save_dir is not None:
            np.save(os.path.join(save_dir, f'CNN_{layer}_metamer_{loss}_ID{ID}_seed{seed}.npy'), input_img.numpy())
            CHANGE_RATE = True

        pbar.set_description(f'loss: {loss}, lr: {lr * lr_factor(step=j, warmup=256, total_steps=n_steps)}, up: {upward_count}')

    return prev_inp, prev_loss


def get_CNN_metamers(sample, orig_model, save_dir, seed):
    input_shape = orig_model.input_shape[1:]
    layers = ['relu1', 'relu2', 'relu3', 'relu4_W', 'relu4_G', 'relu5_W', 'WDrop1', 'fc6_W', 'fc6_G', 'fctop_W', 'fctop_G']
    metamers = [K.variable(np.random.random_sample(sample.shape, )) for _ in range(len(layers))]
    for i,layer in enumerate(layers):
        if seed is not None:
            np.random.seed(int(seed))

        model = keras.Model(
            inputs=orig_model.input,
            outputs=orig_model.get_layer(layer).output
        )

        sample_activation = model.predict(sample)

        input_img = K.variable(np.random.random_sample(sample.shape))
        loss=np.inf
        input_img, loss = optimise_metamer(
            input_img=input_img,
            model=model,
            orig_activation=sample_activation,
            layer=layer,
            n_steps=24000,
            prev_loss=loss,
            save_dir=save_dir,
            seed=seed
        )
        np.save(os.path.join(save_dir, f'CNN_{layer}_metamer_{loss[0]}_ID{ID}_seed{seed}.npy'), input_img.cpu().detach().numpy())
        metamers[i] = input_img
    return metamers

data_path = './testing_data/'
save_path = './eval_mets_CNN/'
seeds = [0, 17, 21, 121, 187, 420, 517, 2137]
#hs = list(range(N_HS))
parser = argparse.ArgumentParser()

parser.add_argument('--samples', nargs='+')
args = parser.parse_args()

for sample_name in args.samples:
    for seed in seeds:
        wav_sample, sr = librosa.load(data_path+sample_name, sr=None)
        ID = sample_name.split('.')[0]

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

        sample = get_cochleagram(wav_sample, sr=sr)
        sample = tf.constant(sample)
        sample = tf.expand_dims(sample, axis=0)
        sample = tf.expand_dims(sample, axis=-1)

        metamers = get_CNN_metamers(sample, model, save_dir=save_path, seed=seed)
        time.sleep(600) # 10 min break after each metamer
