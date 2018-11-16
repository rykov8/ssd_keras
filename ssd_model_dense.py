import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import concatenate

from utils.layers import leaky_relu


def bn_acti_conv(x, filters, kernel_size=1, stride=1, padding='same', activation='relu'):
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    return x

def dense_block(x, n, growth_rate, width=4, activation='relu'):
    input_shape = K.int_shape(x)
    c = input_shape[3]
    for i in range(n):
        x1 = x
        x2 = bn_acti_conv(x, growth_rate*width, 1, 1, activation=activation)
        x2 = bn_acti_conv(x2, growth_rate, 3, 1, activation=activation)
        x = concatenate([x1, x2], axis=3)
        c += growth_rate
    return x

def bl_layer2(x, filters, width, padding='same', activation='relu'):
    x1 = MaxPooling2D(pool_size=2, strides=2, padding=padding)(x)
    x1 = bn_acti_conv(x1, filters, 1, 1, padding, activation=activation)
    x2 = bn_acti_conv(x, filters*width, 1, 1, padding, activation=activation)
    x2 = bn_acti_conv(x2, filters, 3, 2, padding, activation=activation)
    return concatenate([x1, x2], axis=3)


def dsod300_body(x, activation='relu'):
    
    if activation == 'leaky_relu':
        activation = leaky_relu
    
    growth_rate = 48
    compression = 1.0
    source_layers = []
    
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
    # Dense Block 1
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3]*compression), 1, 1, activation=activation)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    # Dense Block 2
    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3]*compression), 1, 1, activation=activation)
    source_layers.append(x) # 38x38x...
    
    x = x2 = MaxPooling2D(pool_size=2, strides=2)(x)
    # Dense Block 3
    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3]*compression), 1, 1, activation=activation)
    # Dense Block 4
    x = dense_block(x, 8, growth_rate, 4, activation)
    
    x1 = bn_acti_conv(x, 256, 1, 1, activation=activation)
    x2 = bn_acti_conv(x2, 256, 1, 1, activation=activation)
    x = concatenate([x1, x2], axis=3)
    source_layers.append(x) # 19x19x512

    x = bl_layer2(x, 256, 1, activation=activation)
    source_layers.append(x) # 10x10x512

    x = bl_layer2(x, 128, 1, activation=activation)
    source_layers.append(x) # 5x5x256

    x = bl_layer2(x, 128, 1, activation=activation)
    source_layers.append(x) # 3x3x256

    x = bl_layer2(x, 128, 1, padding='valid', activation=activation)
    source_layers.append(x) # 1x1x256
    
    return source_layers


def dsod512_body(x, activation='relu'):
    
    if activation == 'leaky_relu':
        activation = leaky_relu
    
    growth_rate = 48
    compression = 1.0
    source_layers = []
    
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
    
    # Dense Block 1
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3]*compression), 1, 1, activation=activation)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    # Dense Block 2
    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3]*compression), 1, 1, activation=activation)
    source_layers.append(x) # 64x64x...

    x = x2 = MaxPooling2D(pool_size=2, strides=2)(x)
    # Dense Block 3
    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3]*compression), 1, 1, activation=activation)
    # Dense Block 4
    x = dense_block(x, 8, growth_rate, 4, activation)
    
    x1 = bn_acti_conv(x, 256, 1, 1, activation=activation)
    x2 = bn_acti_conv(x2, 256, 1, 1, activation=activation)
    x = concatenate([x1, x2], axis=3)
    source_layers.append(x) # 32x32x512

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


def ssd384x512_dense_body(x, activation='relu'):
    # used for SegLink
    
    if activation == 'leaky_relu':
        activation = leaky_relu
    
    growth_rate = 32
    compression = 1.0
    source_layers = []
    
    # Stem
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(96, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3]*compression), 1, 1, activation=activation)
    
    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3]*compression), 1, 1, activation=activation)
    source_layers.append(x)
    
    x = bl_layer2(x, 320, 1, activation=activation)
    source_layers.append(x)

    x = bl_layer2(x, 256, 1, activation=activation)
    source_layers.append(x)

    x = bl_layer2(x, 192, 1, activation=activation)
    source_layers.append(x)

    x = bl_layer2(x, 128, 1, activation=activation)
    source_layers.append(x)
    
    x = bl_layer2(x, 64, 1, activation=activation)
    source_layers.append(x)
    
    return source_layers
