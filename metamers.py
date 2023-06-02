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

N_HS = 13
ID = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data_sample(i):
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate
    sample = dataset[i]["audio"]["array"]
    return sample, sampling_rate


def optimise_metamer(input_img, model, orig_activation, hs_num, n_steps, upward_lim=8, reduce_factor=0.5, prev_loss=None):
    if prev_loss is None:
        prev_loss=np.inf
    upward_count=0

    model.to(device)
    input_img = torch.nn.Parameter(input_img.to(device))
    prev_inp = input_img.detach().clone()
    optimizer = torch.optim.Adam([input_img], lr=1e-1)
    #input_img = input_img.to(device).requires_grad_(True)

    for j in (pbar:= tqdm(range(n_steps))):
        outputs_t = model(input_img)
        hs = torch.square(torch.add(outputs_t.hidden_states[hs_num], -orig_activation[hs_num]))
        loss = torch.mul(torch.norm(hs, dim=(1,2), p=2), 1/(torch.norm(orig_activation[hs_num])+1e-8))

        loss.backward()
        # grads = input_img.grad
        optimizer.step()

        if j==0:
            print(prev_loss)

        if loss[0]==0:
            return input_img, loss[0]

        if loss[0]>prev_loss[0]:
            if upward_count>=upward_lim:
                input_img = torch.nn.Parameter(prev_inp.detach().clone().requires_grad_(True).to(device))
                #input_img = prev_inp.detach().clone().requires_grad_(True)
                upward_count = 0
                optimizer = torch.optim.Adam([input_img], lr=optimizer.param_groups[0]["lr"]*reduce_factor)
            else:
                upward_count += 1
        else:
            upward_count=0
            prev_loss = loss.detach().clone()
            prev_inp = input_img.detach().clone()

        pbar.set_description(f'loss: {loss[0]}, lr: {optimizer.param_groups[0]["lr"]}')
    return prev_inp, prev_loss


def get_AST_metamers(sample, model, save_dir, hidden_states):
    metamers = [torch.tensor(np.random.random_sample(sample.shape), dtype=torch.float32) for _ in range(N_HS)]
    for i in hidden_states:
        input_img = torch.tensor(np.random.random_sample(sample.shape), dtype=torch.float32)
        loss=np.inf
        for _ in range(4):
            input_img, loss = optimise_metamer(
                input_img=input_img,
                model=model,
                orig_activation=model(sample).hidden_states,
                hs_num=i,
                n_steps=256,
                prev_loss=loss
            )
            np.save(os.path.join(save_dir, f'AST_{i}_metamer_{loss[0]}_ID{ID}.npy'), input_img.cpu().detach().numpy())
            print(loss)
        metamers[i] = input_img
    return metamers

parser = argparse.ArgumentParser()

parser.add_argument('--savepath')
parser.add_argument('--hiddenstates', default='all')
args = parser.parse_args()

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

hs = args.hiddenstates
if hs == 'all':
    hs = list(range(N_HS))
else:
    hs = hs.split('-')
    hs = [int(state) for state in hs]

feature_extractor, model = load_AST()
sample = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")['input_values']

model.to(device)
sample = sample.to(device)

metamers = get_AST_metamers(sample, model, save_dir=args.savepath, hidden_states=hs)

plt.figure()
for i in range(N_HS):
    plt.subplot(4,4,i+1)
    librosa.display.specshow(metamers[i].detach().numpy()[0].T)
plt.savefig(os.path.join(args.savepath, 'metamers_plot.png'))



