import tensorflow as tf
import keras.applications
from keras.layers import Conv2D, Embedding, AvgPool2D, Dropout, MaxPool2D, Dense, Layer, Lambda, Flatten, Input


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
    x_w = Dropout(0.1)(x_w)
    x_w = Dense(4096, activation='relu')(x_w)
    x_w = Dropout(0.5)(x_w)
    wout = Dense(wout_shape, activation='softmax')(x_w)

    # genre branch
    x_g = Conv2D(filters=1024, kernel_size=3, strides=1, activation='relu')(x)
    x_g = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(x_g)
    x_g = AvgPool2D(pool_size=(3,3), strides=2)(x_g)
    x_g = Flatten()(x_g)
    x_g = Dropout(0.1)(x_g)
    x_g = Dense(4096, activation='relu')(x_g)
    x_g = Dropout(0.5)(x_g)
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


class Patches(Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size=patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1], # no overlap?
            rates=[1,1,1,1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patch_dims, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch, *args, **kwargs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


