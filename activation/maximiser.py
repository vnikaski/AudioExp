import librosa
import tensorflow as tf
import keras
import keras.backend as K
from tqdm import tqdm
import numpy as np
import pandas as pd

from activation.transforms import padded_jitter


class Maximiser:
    def __init__(self, out_layer_name=None, model=None):
        self.input_shape = None
        self.submodel = None
        self.layer_name = None
        if out_layer_name is not None and model is not None:
            self.init_submodel(model=model, out_layer_name=out_layer_name)

    def init_submodel(self, model: keras.Model, out_layer_name: str) -> keras.Model:
        self.input_shape = model.input_shape[1:]
        sub_model = keras.Model(
            inputs=model.input,
            outputs=model.get_layer(out_layer_name).output
        )
        self.layer_name = out_layer_name
        self.submodel = sub_model

    def _rescale_gradients(self, gradients):
        return gradients/(K.sqrt(tf.reduce_mean(K.square(gradients), axis=[1,2,3], keepdims=True))+1e-6)

    def get_channel_activation(self, channel_number, input_img, mode='mean'):
        layer_output = self.submodel(input_img)
        if mode=='mean':
            return tf.reduce_mean(layer_output[...,channel_number], axis=[1,2])
        else:
            raise NotImplementedError


    def channel(self, channel_number, n_epochs=1000, lr=1e-3, d_t=8, d_f=8, max_cap=None, lr_devcay=None, total_var=1, mode='max', transform_every=1):
        if self.submodel is None:
            raise ValueError('please initialise the submodel first with init_submodel method')

        if mode=='max':
            factor = 1
        elif mode=='min':
            factor = -1
        else:
            raise ValueError(f'Unsupported mode {mode}, please choose "max" or "min"')
        input_img = K.variable(np.random.random_sample((1, *self.input_shape))*31)
        initial_img = tf.identity(input_img)
        every=0
        for _ in (pbar:=tqdm(range(n_epochs))):
            with tf.GradientTape() as gtape:
                mean_activation = self.get_channel_activation(channel_number=channel_number, input_img=input_img, mode='mean')
                total_variation = tf.image.total_variation(input_img)
                max_func = tf.add(mean_activation, -factor*total_var*total_variation)
                grads = gtape.gradient(max_func, input_img)
                pbar.set_description(f"Neuron {channel_number} mean activation: {mean_activation}, objective {max_func}, max_grad: {np.max(grads)}")
            grads = self._rescale_gradients(grads) # todo: check dims with batch
            input_img.assign_add(factor * lr * grads)
            input_img = K.variable(np.clip(input_img, 0, 255))
            every += 1
            if every == transform_every:
                input_img = K.variable(padded_jitter(input_img, d_t, d_f))
                every=0

        return input_img, initial_img

    def look_for_channel(self, channel_number, generator):
        df_most = pd.DataFrame(columns=['layer', 'channel', 'activation', 'fname'])
        df_least = pd.DataFrame(columns=['layer', 'channel', 'activation', 'fname'])

        for i in (pbar:=tqdm(range(len(generator)))):
            X, gen_df = generator.get_data_with_info(i)
            gen_df['activation'] = self.get_channel_activation(channel_number, X).numpy()
            gen_df = gen_df.sort_values('activation').reset_index(drop=True)
            gen_df = gen_df[['activation', 'fname']]
            gen_df['channel'] = [channel_number for _ in range(len(gen_df))]
            gen_df['layer'] = [self.layer_name for _ in range(len(gen_df))]
            df_most = pd.concat([df_most, gen_df.iloc[-5:]], ignore_index=True).reset_index(drop=True).sort_values('activation')
            df_least = pd.concat([df_least, gen_df.iloc[:5]], ignore_index=True).reset_index(drop=True).sort_values('activation')

            df_most = df_most.iloc[-5:].reset_index(drop=True)
            df_least = df_least.iloc[:5].reset_index(drop=True)
            pbar.set_description(f"{channel_number} channel, max activation: {list(df_most['activation'])[-1]}, min activation {df_least['activation'][0]}")
        return df_most, df_least

    def look_for_all_channels(self, generator):
        df_most = pd.DataFrame(columns=['layer', 'channel', 'activation', 'fname'])
        df_least = pd.DataFrame(columns=['layer', 'channel', 'activation', 'fname'])

        for i in (pbar:=tqdm(range(len(generator)))):
            X, gen_df = generator.get_data_with_info(i)
            for channel_number in range(self.submodel.output_shape[-1]):
                gen_df['activation'] = self.get_channel_activation(channel_number, X).numpy()
                gen_df = gen_df.sort_values('activation').reset_index(drop=True)
                gen_df = gen_df[['activation', 'fname']]
                gen_df['channel'] = [channel_number for _ in range(len(gen_df))]
                gen_df['layer'] = [self.layer_name for _ in range(len(gen_df))]
                df_most = pd.concat([df_most, gen_df.iloc[-5:]], ignore_index=True).reset_index(drop=True).sort_values('activation')
                df_least = pd.concat([df_least, gen_df.iloc[:5]], ignore_index=True).reset_index(drop=True).sort_values('activation')

                df_most = pd.concat([
                    df_most[df_most['channel']!=channel_number],
                    df_most[df_most['channel']==channel_number].reset_index(drop=True).iloc[-5:]
                ], ignore_index=True).reset_index(drop=True)
                df_least = pd.concat([
                    df_least[df_least['channel']!=channel_number],
                    df_least[df_least['channel']==channel_number].reset_index(drop=True).iloc[:5]
                ], ignore_index=True).reset_index(drop=True)
                pbar.set_description(f"{channel_number} channel, "
                                     f"max activation: {list(df_most[df_most['channel']==channel_number]['activation'])[-1]}, "
                                     f"min activation {list(df_least[df_least['channel']==channel_number]['activation'])[0]}")
        return df_most, df_least






