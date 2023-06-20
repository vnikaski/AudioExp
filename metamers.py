import torch
import transformers
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
# ID = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rescale_gradients(gradients):
    return gradients/(torch.sqrt(torch.mean(input=torch.square(gradients), dim=[0,1], keepdims=True))+1e-6)

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


def optimise_metamer(input_img, model, orig_activation, hs_num, n_steps, upward_lim=16, reduce_factor=0.5, prev_loss=None, save_dir=None):
    CHANGE_RATE = False
    if prev_loss is None:
        prev_loss=np.full(input_img.shape[0], fill_value=np.inf)
    upward_count=np.full(input_img.shape[0], fill_value=0)

    #input_img = torch.nn.Parameter(input_img.to(device))
    input_img = torch.nn.Parameter(input_img.detach().clone().requires_grad_(True).to(device))
    prev_inp = input_img.detach().clone()
    lr = 0.001
    #optimizer = torch.optim.Adam([input_img], lr=1e-3)
    #scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=256, num_training_steps=n_steps)
    #input_img = input_img.to(device).requires_grad_(True)

    for j in (pbar:= tqdm(range(n_steps))):
        outputs_t = model(input_img)
        hs = torch.add(outputs_t.hidden_states[hs_num], -orig_activation[hs_num])
        loss = torch.mul(torch.norm(hs, dim=(1,2), p=2), 1/(torch.norm(orig_activation[hs_num], p=2)+1e-8))

        lfac = lr_factor(step=j, warmup=256, total_steps=n_steps)
        new_input_img = input_img.detach().clone()
        for m in range(len(loss)):
            if m!=len(loss)-1:
                loss[m].backward(inputs=input_img, retain_graph=True)
            else:
                loss[m].backward(inputs=input_img, retain_graph=False)
            grads = input_img.grad
            with torch.no_grad():
                new_input_img[m] -= lr * lfac * rescale_gradients(grads[m])
        #optimizer.step()
        """"
        if loss[0]==0:
            return input_img, loss[0]
        """
        for m in range(len(loss)):
            if loss[m] >= prev_loss[m]:
                if upward_count[m] >= upward_lim:
                    with torch.no_grad():
                        input_img[m] = prev_inp[m].detach().clone().requires_grad_(True)
                    #input_img = prev_inp.detach().clone().requires_grad_(True)
                    upward_count[m] = 0
                else:
                    upward_count[m] += 1
            else:
                upward_count[m]=0
                prev_loss[m] = loss[m].detach().clone()
                prev_inp[m] = input_img[m].detach().clone()

        input_img = torch.Tensor(np.clip(input_img.detach().cpu().numpy(), a_min=-1.5, a_max=1.5))
        input_img = torch.nn.Parameter(input_img.requires_grad_(True).to(device))

        if j%500 == 0 and save_dir is not None:
            for k in range(input_img.shape[0]):
                np.save(os.path.join(save_dir, f'{k}_AST_{hs_num}_metamer_{loss[0]}_ID{ID}.npy'), input_img.cpu().detach().numpy())
            CHANGE_RATE = True

        #pbar.set_description(f'loss: {loss[0]}, lr: {optimizer.param_groups[0]["lr"]}, up: {upward_count}')
        pbar.set_description(f'loss: {["{0:.4f}".format(loss[m]) for m in range(len(loss))]}, lr: {lr * lr_factor(step=j, warmup=256, total_steps=n_steps)}, up: {upward_count}')

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


def get_AST_metamers(sample, model, save_dir, hidden_states, n_mets=1):
    metamers = [torch.tensor(np.random.random_sample((n_mets, *sample.shape[1:])), dtype=torch.float32) for _ in range(N_HS)]
    sample_activation = model(sample).hidden_states
    for i in hidden_states:
        input_img = torch.tensor(np.random.random_sample((n_mets, *sample.shape[1:])), dtype=torch.float32)
        loss=np.full(input_img.shape[0], fill_value=np.inf)
        input_img, loss = optimise_metamer(
            input_img=input_img,
            model=model,
            orig_activation=sample_activation,
            hs_num=i,
            n_steps=24000,
            prev_loss=loss,
            save_dir=save_dir
        )
        np.save(os.path.join(save_dir, f'AST_{i}_metamers_{loss[0]}_ID{ID}.npy'), input_img.cpu().detach().numpy())
        metamers[i] = input_img
    return metamers

parser = argparse.ArgumentParser()

parser.add_argument('--savepath')
parser.add_argument('--hiddenstates', default='all')
parser.add_argument('--id', default=0)
parser.add_argument('--nmets', default=1)
args = parser.parse_args()

ID = int(args.id)

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
sample = feature_extractor(dataset[ID]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")['input_values']

model.to(device)
for param in model.parameters():
    param.requires_grad = False
sample = sample.to(device)

metamers = get_AST_metamers(sample, model, save_dir=args.savepath, hidden_states=hs, n_mets=int(args.nmets))

plt.figure()
for i in range(N_HS):
    plt.subplot(4,4,i+1)
    librosa.display.specshow(metamers[i].cpu().detach().numpy()[0].T)
plt.savefig(os.path.join(args.savepath, 'metamers_plot.png'))



