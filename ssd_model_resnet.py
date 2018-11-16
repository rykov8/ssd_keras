import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import add
from keras.regularizers import l2

kernel_initializer = 'he_normal' 
kernel_regularizer = l2(1.e-4)

def _shortcut(x, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1,1),
                          strides=(stride_width, stride_height), padding="valid", 
                          kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    else:
        shortcut = x
    return add([shortcut, residual])

def _bn_relu_conv(x, filters, kernel_size, strides=(1,1), padding="same"):
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, 
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    return x

def bl_bottleneck(x, filters, strides=(1,1), is_first_layer_of_first_block=False):
    if is_first_layer_of_first_block:
        # don't repeat bn->relu since we just did bn->relu->maxpool
        x1 = Conv2D(filters, (1,1), strides=strides, padding="same", 
                    kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    else:
        x1 = _bn_relu_conv(x, filters=filters, kernel_size=(1,1), strides=strides)
    x1 = _bn_relu_conv(x1, filters=filters, kernel_size=(3,3))
    x1 = _bn_relu_conv(x1, filters=filters*4, kernel_size=(1,1))
    return _shortcut(x, x1)


def ssd512_resnet_body(x, activation='relu'):
    
    if activation == 'leaky_relu':
        activation = leaky_relu
    
    source_layers = []
    
    x = Conv2D(64, (7,7), strides=(2,2), padding='same', 
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    x = bl_bottleneck(x, filters=64, is_first_layer_of_first_block=True)
    x = bl_bottleneck(x, filters=64)
    x = bl_bottleneck(x, filters=64)

    x = bl_bottleneck(x, filters=128, strides=(2,2))
    x = bl_bottleneck(x, filters=128)
    x = bl_bottleneck(x, filters=128)
    x = bl_bottleneck(x, filters=128)
    source_layers.append(x)
    
    x = bl_bottleneck(x, filters=256, strides=(2,2))
    x = bl_bottleneck(x, filters=256)
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)
    
    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)
    
    return source_layers


