import tensorflow as tf
import keras.applications
from keras.layers import Conv2D, AvgPool2D, MaxPool2D, Dense, Lambda, Flatten, Input


def Kell2018(input_shape, wout_shape, gout_shape):
    # shared pathway
    inp = Input(input_shape)
    x = Conv2D(filters=96, kernel_size=9, strides=3, input_shape=input_shape, activation='relu')(inp)
    x = MaxPool2D(pool_size=(3,3), strides=2)(x)
    x = Lambda(tf.nn.local_response_normalization)(x)
    x = Conv2D(filters=256, kernel_size=5, strides=2, activation='relu')(x)
    x = MaxPool2D(pool_size=(3,3), strides=2)(x)
    x = Lambda(tf.nn.local_response_normalization)(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(x)

    # word branch
    x_w = Conv2D(filters=1024, kernel_size=3, strides=1, activation='relu')(x)
    x_w = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(x_w)
    x_w = AvgPool2D(pool_size=(3,3), strides=2)(x_w)
    x_w = Flatten()(x_w)
    x_w = Dense(4096, activation='relu')(x_w)
    wout = Dense(wout_shape, activation='softmax')(x_w)

    # genre branch
    x_g = Conv2D(filters=1024, kernel_size=3, strides=1, activation='relu')(x)
    x_g = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(x_g)
    x_g = AvgPool2D(pool_size=(3,3), strides=2)(x_g)
    x_g = Flatten()(x_g)
    x_g = Dense(4096, activation='relu')(x_g)
    gout = Dense(gout_shape, activation='softmax')(x_g)

    model = keras.Model(inp, [wout, gout])
    model.output_names = ['wout', 'gout']

    return model

def Kell2018small(input_shape, wout_shape, gout_shape):
    # shared pathway
    inp = Input(input_shape)
    x = Conv2D(filters=64, kernel_size=9, strides=3, input_shape=input_shape, activation='relu')(inp)
    x = MaxPool2D(pool_size=(3,3), strides=2)(x)
    x = Lambda(tf.nn.local_response_normalization)(x)
    x = Conv2D(filters=128, kernel_size=5, strides=2, activation='relu')(x)
    x = MaxPool2D(pool_size=(3,3), strides=2)(x)
    x = Lambda(tf.nn.local_response_normalization)(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu')(x)

    # word branch
    x_w = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(x)
    x_w = Conv2D(filters=256, kernel_size=3, strides=1, activation='relu')(x_w)
    x_w = AvgPool2D(pool_size=(3,3), strides=2)(x_w)
    x_w = Flatten()(x_w)
    x_w = Dense(2048, activation='relu')(x_w)
    wout = Dense(wout_shape, activation='softmax')(x_w)

    # genre branch
    x_g = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(x)
    x_g = Conv2D(filters=256, kernel_size=3, strides=1, activation='relu')(x_g)
    x_g = AvgPool2D(pool_size=(3,3), strides=2)(x_g)
    x_g = Flatten()(x_g)
    x_g = Dense(2048, activation='relu')(x_g)
    gout = Dense(gout_shape, activation='softmax')(x_g)

    model = keras.Model(inp, [wout, gout])
    model.output_names = ['wout', 'gout']

    return model
