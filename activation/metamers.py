import torch
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

HS_NUMS = list(range(13))
ID = 0

def get_data_sample(i):
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate
    sample = dataset[i]["audio"]["array"]
    return sample, sampling_rate


def optimise_metamer(input_img, model, orig_activation, hs_num, n_steps, upward_lim=8, reduce_factor=0.5):
    input_img.requires_grad_(True)
    optimizer = torch.optim.Adam([input_img], lr=1e-1)
    prev_loss=np.inf
    prev_inp= input_img.detach().clone()
    upward_count=0

    for _ in (pbar:= tqdm(range(n_steps))):
        outputs_t = model(input_img)
        hs = torch.square(torch.add(outputs_t.hidden_states[hs_num], -orig_activation[hs_num]))
        loss = torch.mul(torch.norm(hs, dim=(1,2), p=2), 1/(torch.norm(orig_activation[hs_num])+1e-8))

        loss.backward()
        grads = input_img.grad
        optimizer.step()

        if loss==0:
            return input_img, loss[0]

        if loss>prev_loss:
            if upward_count>=upward_lim:
                input_img = prev_inp.detach().clone().requires_grad_(True)
                upward_count = 0
                optimizer = torch.optim.Adam([input_img], lr=optimizer.param_groups[0]["lr"]*reduce_factor)
            else:
                upward_count += 1
        else:
            prev_loss = loss.detach().clone()
            prev_inp = input_img.detach().clone()

        pbar.set_description(f'loss: {loss[0]}, lr: {optimizer.param_groups[0]["lr"]}')
    return prev_inp, prev_loss[0]


def get_AST_metamers(sample, model, save_dir):
    metamers = [torch.tensor(np.random.random_sample(sample.shape), dtype=torch.float32, requires_grad=True) for i in range(13)]
    for i in HS_NUMS:
        input_img = torch.tensor(np.random.random_sample(sample.shape), dtype=torch.float32, requires_grad=True)
        for _ in range(4):
            input_img, loss = optimise_metamer(
                input_img=input_img,
                model=model,
                orig_activation=model(sample).hidden_states,
                hs_num=i,
                n_steps=6000,
            )
        metamers[i] = input_img
        np.save(os.path.join(save_dir, f'AST_{i}_metamer_{loss}_ID{ID}.npy'))
    return metamers

parser = argparse.ArgumentParser()

parser.add_argument('--savepath')
args = parser.parse_args()

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor, model = load_AST()
sample = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")['input_values']

metamers = get_AST_metamers(sample, model, save_dir=args.savepath)

plt.figure()
for i in range(13):
    plt.subplot(4,4,i+1)
    librosa.display.specshow(metamers[i].detach().numpy()[0].T)
plt.savefig(os.path.join(args.savepath, 'metamers_plot.png'))



