import tensorflow as tf
import keras.applications
from keras.layers import Conv2D, Embedding, ReLU, AvgPool2D, Dropout, Concatenate, Reshape, Permute, Add, MultiHeadAttention, MaxPool2D, Dense, Layer, Lambda, Flatten, Input, LayerNormalization
import keras.backend as K
import numpy as np

def Kell2018(input_shape, wout_shape, gout_shape, pretrained=True):
    # shared pathway
    inp = Input(input_shape)
    x = Conv2D(filters=96, kernel_size=9, strides=3, input_shape=input_shape, activation=None, name='conv1', padding='same')(inp)
    x = ReLU(name='relu1')(x)
    x = MaxPool2D(pool_size=(3,3), strides=2, name='max_pool1', padding='same')(x)
    x = Lambda(tf.nn.local_response_normalization, arguments={'depth_radius': 5, 'bias': 1, 'alpha': 1e-3, 'beta': 0.75},name='LRN1')(x)
    x = Conv2D(filters=256, kernel_size=5, strides=2, activation=None, name='conv2', padding='same')(x)
    x = ReLU(name='relu2')(x)
    x = MaxPool2D(pool_size=(3,3), strides=2, name='max_pool2', padding='same')(x)
    x = Lambda(tf.nn.local_response_normalization, arguments={'depth_radius': 5, 'bias': 1, 'alpha': 1e-3, 'beta': 0.75}, name='LRN2')(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, activation=None, name='conv3', padding='same')(x)
    x = ReLU(name='relu3')(x)

    # word branch
    x_w = Conv2D(filters=1024, kernel_size=3, strides=1, activation=None, name='conv4_W', padding='same')(x)
    x_w = ReLU(name='relu4_W')(x_w)
    x_w = Conv2D(filters=512, kernel_size=3, strides=1, activation=None, name='conv5_W', padding='same')(x_w)
    x_w = ReLU(name='relu5_W')(x_w)
    x_w = AvgPool2D(pool_size=(3,3), strides=2, name='avg_pool5_W', padding='same')(x_w)
    x_w = Flatten(name='WFlatten')(x_w)
    x_w = Dropout(0.1, name='WDrop1')(x_w)
    x_w = Dense(1024, activation='relu', name='fc6_W')(x_w)
    x_w = Dropout(0.5, name='WDrop2')(x_w)
    wout = Dense(wout_shape, name='fctop_W', activation='softmax')(x_w)

    # genre branch
    x_g = Conv2D(filters=1024, kernel_size=3, strides=1, activation=None, name='conv4_G', padding='same')(x)
    x_g = ReLU(name='relu4_G')(x_g)
    x_g = Conv2D(filters=512, kernel_size=3, strides=1, activation=None, name='conv5_G', padding='same')(x_g)
    x_g = ReLU(name='relu5_G')(x_g)
    x_g = AvgPool2D(pool_size=(3,3), strides=2, name='avg_pool5_G', padding='same')(x_g)
    x_g = Flatten(name='GFlatten')(x_g)
    x_g = Dropout(0.1, name='GDrop1')(x_g)
    x_g = Dense(1024, activation='relu', name='fc6_G')(x_g)
    x_g = Dropout(0.5, name='GDrop2')(x_g)
    gout = Dense(gout_shape, name='fctop_G', activation='softmax')(x_g)

    model = keras.Model(inp, [wout, gout])
    model.output_names = ['wout', 'gout']

    if pretrained==True:
        weights = np.load('models/network_weights_early_layers.npy', allow_pickle=True, encoding='latin1') # https://github.com/mcdermottLab/kelletal2018/blob/master/network/weights/network_weights_early_layers.npy
        weights = weights.item()
        for lname in list(weights.keys()):
            model.get_layer(lname).set_weights([
                weights[lname]['W'],
                weights[lname]['b']
            ])
    return model

def Kell2018small(input_shape, wout_shape, gout_shape):
    # shared pathway
    inp = Input(input_shape)
    x = Conv2D(filters=64, kernel_size=9, strides=3, input_shape=input_shape, activation='relu', name='Conv1')(inp)
    x = MaxPool2D(pool_size=(3,3), strides=2, name='MaxPool1')(x)
    x = Lambda(tf.nn.local_response_normalization, name='LRN1')(x)
    x = Conv2D(filters=128, kernel_size=5, strides=2, activation='relu', name='Conv2')(x)
    x = MaxPool2D(pool_size=(3,3), strides=2, name='MaxPool2')(x)
    x = Lambda(tf.nn.local_response_normalization, name='LRN2')(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', name='Conv3')(x)

    # word branch
    x_w = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', name='WConv1')(x)
    x_w = Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', name='WConv2')(x_w)
    x_w = AvgPool2D(pool_size=(3,3), strides=2, name='WAvgPool1')(x_w)
    x_w = Flatten(name='WFlatten')(x_w)
    x_w = Dropout(0.1, name='WDrop1')(x_w)
    x_w = Dense(2048, activation='relu', name='WDense1')(x_w)
    x_w = Dropout(0.5, name='WDrop2')(x_w)
    wout = Dense(wout_shape, activation='softmax', name='WDense2')(x_w)

    # genre branch
    x_g = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', name='GConv1')(x)
    x_g = Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', name='GConv2')(x_g)
    x_g = AvgPool2D(pool_size=(3,3), strides=2, name='GAvgPool1')(x_g)
    x_g = Flatten(name='GFlatten')(x_g)
    x_g = Dropout(0.1, name='GDrop1')(x_g)
    x_g = Dense(2048, activation='relu', name='GDense1')(x_g)
    x_g = Dropout(0.5, name='GDrop2')(x_g)
    gout = Dense(gout_shape, activation='softmax', name='GDense2')(x_g)

    model = keras.Model(inp, [wout, gout])
    model.output_names = ['wout', 'gout']

    return model

class Patches(Layer):
    def __init__(self, patch_size, overlap):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap

    def call(self, images, *args, **kwargs):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0]-self.overlap[0], self.patch_size[1]-self.overlap[1], 1], # no overlap?
            rates=[1,1,1,1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch-size': self.patch_size,
        })


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.d = projection_dim

        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch, *args, **kwargs):
        positions = tf.range(start=1, limit=self.num_patches+1, delta=1) # 0 reserved for CLS token
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            'num-patches': self.num_patches,
            'd': self.d
        })


class PatchEmbed(Layer):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, input_shape=(*img_size, in_chans))
        self.reshape = Reshape(self.num_patches, embed_dim)
        self.permute = Permute((2,1))

    def call(self, x, *args, **kwargs):
        x = self.proj(x) # (batch, patch_pos1, patch_pos2, embed_dim)
        x = self.reshape(x) # (batch, num_patches, embed_dim)
        x = self.permute(x) # (batch, embed_dim, num_patches)
        return x

class CLSConcat(Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.concat = Concatenate(axis=1)

    def build(self, input_shape):
        self.cls = self.add_weight(name='cls',
                                   shape=(1,1,self.dim),
                                   initializer='normal',
                                   trainable=True)
        assert self.dim == input_shape[-1]
        super(CLSConcat, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch_size = tf.shape(inputs)[0]
        cls_batch = K.tile(x=self.cls, n=(batch_size, 1, 1))
        return self.concat([cls_batch, inputs])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, self.dim)




class MLP(Layer):
    def __init__(self, units, rate):
        super(MLP, self).__init__()
        self.units = units
        self.rate = rate
        self.layers = [[Dense(unit, activation='gelu'), Dropout(rate)] for unit in units]

    def call(self, x, *args, **kwargs):
        for layers in self.layers:
            for layer in layers:
                x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'rate': self.rate
        })
        return config


class Transformer(Layer):
    def __init__(self, num_heads, key_dims, hidden_units):
        super().__init__()
        self.heads = num_heads
        self.key_dims = key_dims
        self.hidden_units = hidden_units

        self.norm = LayerNormalization(epsilon=1e-6)
        self.MHA = MultiHeadAttention(num_heads=num_heads, key_dim=key_dims, dropout=0.1)
        self.mlp = MLP(units=[hidden_units], rate=0.1)
        self.add = Add()

    def call(self, X, *args, **kwargs):
        inputs = X
        x = X
        x = self.norm(x)
        x = self.MHA(x,x)
        y = self.add([x, inputs])
        x = self.norm(y)
        x = self.mlp(x)
        x = self.add([x,y])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'L': self.L,
            'heads': self.heads,
            'key_dims': self.key_dims,
            'hidden_units': self.hidden_units
        })
        return config

def ASTClassifierConnected(input_shape, patch_size, overlap, projection_dims, num_heads, hidden_units, n_transformer_layers, mlp_head_units, wout_classes, gout_classes):
    num_patches = ((input_shape[1]-overlap[1]) // (patch_size[1]-overlap[1])) * ((input_shape[0]-overlap[0]) // (patch_size[0]-overlap[0]))

    inputs = Input(shape=input_shape)
    x = Patches(patch_size=patch_size,overlap=overlap)(inputs)
    x = PatchEncoder(num_patches=num_patches, projection_dim=projection_dims)(x)
    x = CLSConcat(dim=projection_dims)(x)
    # x = PatchEmbed(input_shape[:2], embed_dim=projection_dims, patch_size=patch_size, in_chans=input_shape[-1])
    for i in range(n_transformer_layers):
        x = Transformer(num_heads=num_heads, key_dims=projection_dims, hidden_units=hidden_units)(x)
    representation = LayerNormalization(epsilon=1e-6)(x)
    cls_out = representation[:,0,:]
    cls_out = Flatten()(cls_out)
    cls_out = Dropout(0.5)(cls_out)
    # wout head
    wout_features = MLP([mlp_head_units], rate=0.5)(cls_out)
    wout = Dense(wout_classes, activation='softmax')(wout_features)
    # gout head
    gout_features = MLP([mlp_head_units], rate=0.5)(cls_out)
    gout = Dense(gout_classes, activation='softmax')(gout_features)

    model = keras.Model(inputs=inputs, outputs=[wout, gout])
    model.output_names = ['wout', 'gout']
    return model



