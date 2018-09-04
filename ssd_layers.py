"""Some special pupropse layers for SSD."""

import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def leaky_relu(x):
    """Leaky Rectified Linear activation.
    
    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """
    #return K.relu(x, alpha=0.1, max_value=None)
    
    # requires less memory then keras implementation
    alpha = 0.1
    zero = _to_tensor(0., x.dtype.base_dtype)
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x = alpha * tf.minimum(x, zero) + tf.maximum(x, zero)
    return x


class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    #TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name=self.name+'_gamma')
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output

