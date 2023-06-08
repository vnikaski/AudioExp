import tensorflow as tf
import keras
import keras.backend as K
from tqdm import tqdm
import pandas as pd
import numpy as np
import keras
import sklearn
import librosa
import librosa.display
from matplotlib import pyplot as plt
import os
from datasets import load_dataset
import argparse

from models.load_AST import load_AST
from models.models import Kell2018
from preprocessing.spectrograms import get_cochleagram

N_HS = 13
# ID = 1


def rescale_gradients(gradients):
    return gradients/(K.sqrt(K.mean(K.square(gradients), axis=[1,2], keepdims=True))+1e-6)

"""
def lr_factor(step, warmup, total_steps):
    if step < warmup:
        return step * (1/warmup)
    elif step >= warmup:
        return 1 - (step-warmup)*(1/(total_steps-warmup))
"""

def lr_factor(step, warmup, total_steps):
    if step < warmup:
        return step*(1/warmup)
    elif step >= warmup:
        return 0.5 * (1 + np.cos(np.pi * (step-warmup) / (total_steps-warmup)))


def get_data_sample(i):
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate
    sample = dataset[i]["audio"]["array"]
    return sample, sampling_rate


def optimise_metamer(input_img, model, orig_activation, hs_num, n_steps, upward_lim=8, reduce_factor=0.5, prev_loss=None, save_dir=None):
    CHANGE_RATE = False
    if prev_loss is None:
        prev_loss=np.inf
    upward_count=0

    #input_img = torch.nn.Parameter(input_img.to(device))
    input_img = K.variable(input_img)
    prev_inp = tf.identity(input_img)
    lr = 0.001
    #optimizer = torch.optim.Adam([input_img], lr=1e-3)
    #scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=256, num_training_steps=n_steps)
    #input_img = input_img.to(device).requires_grad_(True)

    for j in (pbar:= tqdm(range(n_steps))):
        with tf.GradientTape() as gtape:
            outputs_t = model(input_img)
            hs = tf.add(outputs_t,-orig_activation)
            loss = tf.math.multiply(tf.norm(hs, ord=2), 1/(tf.norm(orig_activation, ord=2)+1e-8))

            grads = gtape.gradient(loss, input_img)

        input_img.assign_add(- lr * lr_factor(step=j, warmup=256, total_steps=n_steps) * rescale_gradients(grads))

        if loss==0:
            return input_img, loss

        if loss>prev_loss:
            if upward_count>=upward_lim:
                input_img = K.variable(prev_inp)
                upward_count = 0
            else:
                upward_count += 1
        else:
            upward_count=0
            prev_loss = tf.identity(loss)
            prev_inp = tf.identity(input_img)

        input_img = K.variable(np.clip(input_img.numpy(), a_min=-1.5, a_max=1.5))

        if j%6000 == 0 and save_dir is not None:
            np.save(os.path.join(save_dir, f'CNN_{hs_num}_metamer_{loss}_ID{ID}.npy'), input_img.numpy())
            CHANGE_RATE = True

        #pbar.set_description(f'loss: {loss[0]}, lr: {optimizer.param_groups[0]["lr"]}, up: {upward_count}')
        pbar.set_description(f'loss: {loss}, lr: {lr * lr_factor(step=j, warmup=256, total_steps=n_steps)}, up: {upward_count}')

    """
    for j in (pbar:= tqdm(range(n_steps))):
        outputs_t = model(input_img)
        hs = torch.square(torch.add(outputs_t.hidden_states[hs_num], -orig_activation[hs_num]))
        loss = torch.mul(torch.norm(hs, dim=(1,2), p=2), 1/(torch.norm(orig_activation[hs_num])+1e-8))

        loss.backward()
        # grads = input_img.grad
        optimizer.step()

        if loss[0]==0:
            return input_img, loss[0]

        if loss>prev_loss:
            if upward_count>=upward_lim:
                input_img = torch.nn.Parameter(prev_inp.detach().clone().requires_grad_(True).to(device))
                #input_img = prev_inp.detach().clone().requires_grad_(True)
                upward_count = 0
                if CHANGE_RATE:
                    optimizer = torch.optim.Adam([input_img], lr=0.1)
                    CHANGE_RATE = False
                else:
                    optimizer = torch.optim.Adam([input_img], lr=optimizer.param_groups[0]["lr"]*reduce_factor)
            else:
                upward_count += 1
        else:
            upward_count=0
            prev_loss = loss.detach().clone()
            prev_inp = input_img.detach().clone()

        if j%6000 == 0 and save_dir is not None:
            np.save(os.path.join(save_dir, f'AST_{hs_num}_metamer_{loss[0]}_ID{hs_num}.npy'), input_img.cpu().detach().numpy())
            CHANGE_RATE = True

        pbar.set_description(f'loss: {loss[0]}, lr: {optimizer.param_groups[0]["lr"]}, up: {upward_count}')
    """
    return prev_inp, prev_loss


def get_CNN_metamers(sample, model, save_dir, hidden_states):
    metamers = [K.variable(np.random.random_sample(sample.shape)) for _ in range(N_HS)]
    for i, layer_name in enumerate(hidden_states):
        submodel = keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        sample_activation = submodel(sample)
        input_img = K.variable(np.random.random_sample(sample.shape)*63)
        loss=np.inf
        input_img, loss = optimise_metamer(
            input_img=input_img,
            model=submodel,
            orig_activation=sample_activation,
            hs_num=layer_name,
            n_steps=24000,
            prev_loss=loss,
            save_dir=save_dir
        )
        np.save(os.path.join(save_dir, f'CNN_{layer_name}_metamer_{loss}_ID{ID}.npy'), input_img.numpy())
        metamers[i] = input_img
    return metamers

parser = argparse.ArgumentParser()

parser.add_argument('--savepath')
parser.add_argument('--hiddenstates', default='all')
parser.add_argument('--id', default=0)
args = parser.parse_args()

ID = int(args.id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

hs = args.hiddenstates
if hs == 'all':
    hs = ['conv1', 'conv2', 'conv3', 'conv4_W', 'conv4_G', 'conv5_W', 'conv5_G', 'fc6_W', 'fc6_G']
else:
    hs = hs.split('-')
    hs = [int(state) for state in hs]

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

sample = dataset[ID]["audio"]["array"]
if len(sample) >= int(sampling_rate*2):
    sample = sample[:int(sampling_rate*2)]
else:
    sample = librosa.util.pad_center(sample, size=int(sampling_rate*2))
sample = get_cochleagram(sample, sampling_rate)
sample = np.reshape(sample, (1, *sample.shape, 1))

metamers = get_CNN_metamers(sample, model, save_dir=args.savepath, hidden_states=hs)



