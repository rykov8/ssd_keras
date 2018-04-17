"""Keras implementation of SSOD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Model

from ssd_layers import Normalize
from ssd_layers import leaky_relu
from ssd_model import multibox_head

# keras default inialization is glorot == xavier, use glorot_normal?
# kernel_initializer='glorot_uniform', bias_initializer='zeros'


def bn_acti_conv(x, filters, kernel_size, stride, padding='same', activation='relu'):
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    #x = Dropout(0.5)(x)
    return x

def bl_layer1(x, filters, width, activation='relu'):
    x1 = x
    x2 = bn_acti_conv(x, filters*width, 1, 1, activation=activation)
    x2 = bn_acti_conv(x2, filters, 3, 1, activation=activation)
    return concatenate([x1, x2], axis=3)

def bl_layer2(x, filters, width, padding='same', activation='relu'):
    x1 = MaxPooling2D(pool_size=2, strides=2, padding=padding)(x)
    x1 = bn_acti_conv(x1, filters, 1, 1, padding, activation=activation)
    x2 = bn_acti_conv(x, filters*width, 1, 1, padding, activation=activation)
    x2 = bn_acti_conv(x2, filters, 3, 2, padding, activation=activation)
    return concatenate([x1, x2], axis=3)


def dsod300_body(x, activation='relu'):
    
    if activation == 'leaky_relu':
        activation = leaky_relu
    
    # Stem
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    
    growth_rate = 48
    num_channels = 128
    source_layers = []

    for i in range(6):
        x = bl_layer1(x, growth_rate, 4, activation=activation)
        num_channels += growth_rate

    x = bn_acti_conv(x, num_channels, 1, 1, activation=activation)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    for i in range(8):
        x = bl_layer1(x, growth_rate, 4, activation=activation)
        num_channels += growth_rate

    x = bn_acti_conv(x, num_channels, 1, 1, activation=activation)
    source_layers.append(x) # 38x38x512

    x = MaxPooling2D(pool_size=2, strides=2)(x)

    for i in range(8):
        x = bl_layer1(x, growth_rate, 4, activation=activation)
        num_channels += growth_rate

    x = bn_acti_conv(x, num_channels, 1, 1, activation=activation)

    for i in range(8):
        x = bl_layer1(x, growth_rate, 4, activation=activation)
        num_channels += growth_rate

    x1 = bn_acti_conv(x, 256, 1, 1, activation=activation)

    x2 = MaxPooling2D(pool_size=2, strides=2)(source_layers[0])
    x2 = bn_acti_conv(x2, 256, 1, 1, activation=activation)
    x = concatenate([x1, x2], axis=3)
    source_layers.append(x) # 19x19x1024

    x = bl_layer2(x, 256, 1, activation=activation)
    source_layers.append(x) # 10x10x512

    x = bl_layer2(x, 128, 1, activation=activation)
    source_layers.append(x) # 5x5x256

    x = bl_layer2(x, 128, 1, activation=activation)
    source_layers.append(x) # 3x3x256

    x = bl_layer2(x, 128, 1, 'valid', activation=activation)
    source_layers.append(x) # 1x1x256
    
    return source_layers


def dsod512_body(x, activation='relu'):
    
    if activation == 'leaky_relu':
        activation = leaky_relu
    
    # Stem
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    
    growth_rate = 48
    num_channels = 128
    source_layers = []

    for i in range(6):
        x = bl_layer1(x, growth_rate, 4)
        num_channels += growth_rate

    x = bn_acti_conv(x, num_channels, 1, 1, activation=activation)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    for i in range(8):
        x = bl_layer1(x, growth_rate, 4)
        num_channels += growth_rate

    x = bn_acti_conv(x, num_channels, 1, 1, activation=activation)
    source_layers.append(x) # 64x64x512

    x = MaxPooling2D(pool_size=2, strides=2)(x)

    for i in range(8):
        x = bl_layer1(x, growth_rate, 4)
        num_channels += growth_rate

    x = bn_acti_conv(x, num_channels, 1, 1, activation=activation)

    for i in range(8):
        x = bl_layer1(x, growth_rate, 4)
        num_channels += growth_rate

    x1 = bn_acti_conv(x, 256, 1, 1, activation=activation)

    x2 = MaxPooling2D(pool_size=2, strides=2)(source_layers[0])
    x2 = bn_acti_conv(x2, 256, 1, 1, activation=activation)
    x = concatenate([x1, x2], axis=3)
    source_layers.append(x) # 32x32x1024

    x = bl_layer2(x, 256, 1, activation=activation)
    source_layers.append(x) # 16x16x512

    x = bl_layer2(x, 128, 1, activation=activation)
    source_layers.append(x) # 8x8x256

    x = bl_layer2(x, 128, 1, activation=activation)
    source_layers.append(x) # 4x4x256

    x = bl_layer2(x, 128, 1, activation=activation)
    source_layers.append(x) # 2x2x256
    
    x = bl_layer2(x, 128, 1, activation=activation)
    source_layers.append(x) # 1x1x256
    
    return source_layers


def DSOD300(input_shape=(300, 300, 3), num_classes=21, activation='relu'):
    
    K.clear_session()
    
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod300_body(x, activation=activation)

    num_priors = [4, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20]

    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations)
    model = Model(input_tensor, output_tensor)

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.source_layers_names = [l.name.split('/')[0] for l in source_layers]
    model.aspect_ratios = [[1,2], [1,2,3], [1,2,3], [1,2,3], [1,2], [1,2]]
    model.minmax_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
    model.steps = [8, 16, 32, 64, 100, 300]

    return model


def DSOD512(input_shape=(512, 512, 3), num_classes=21, activation='relu'):
    
    K.clear_session()
    
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod512_body(x, activation=activation)

    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20, 20]

    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations)
    model = Model(input_tensor, output_tensor)

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.source_layers_names = [l.name.split('/')[0] for l in source_layers]
    model.aspect_ratios = [[1,2], [1,2,3], [1,2,3], [1,2,3], [1,2,3], [1,2], [1,2]]
    model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460), (460, 537)]
    model.steps = [8, 16, 32, 64, 128, 256, 512]

    return model
