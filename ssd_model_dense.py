import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.layers import Dropout

from ssd_layers import leaky_relu


def bn_acti_conv(x, filters, kernel_size=1, stride=1, padding='same', activation='relu'):
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

