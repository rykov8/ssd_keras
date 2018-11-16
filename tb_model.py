"""Keras implementation of TextBoxes."""

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
from keras.models import Model

from utils.layers import Normalize
from ssd_model import ssd300_body


def multibox_head(source_layers, num_priors, num_classes, normalizations=None, softmax=True):

    mbox_conf = []
    mbox_loc = []
    for i in range(len(source_layers)):
        x = source_layers[i]
        name = x.name.split('/')[0]
        
        # normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + '_norm'
            x = Normalize(normalizations[i], name=name)(x)
            
        # confidence
        name1 = name + '_mbox_conf'
        x1 = Conv2D(num_priors[i] * num_classes, (1, 5), padding='same', name=name1)(x)
        x1 = Flatten(name=name1+'_flat')(x1)
        mbox_conf.append(x1)

        # location
        name2 = name + '_mbox_loc'
        x2 = Conv2D(num_priors[i] * 4, (1, 5), padding='same', name=name2)(x)
        x2 = Flatten(name=name2+'_flat')(x2)
        mbox_loc.append(x2)

    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    if softmax:
        mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)
    else:
        mbox_conf = Activation('sigmoid', name='mbox_conf_final')(mbox_conf)

    predictions = concatenate([mbox_loc, mbox_conf], axis=2, name='predictions')
    
    return predictions


def TB300(input_shape=(300, 300, 3), num_classes=2, softmax=True):
    """TextBoxes300 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.
    
    # References
        - [TextBoxes: A Fast Text Detector with a Single Deep Neural Network](https://arxiv.org/abs/1611.06779)
    """
    
    K.clear_session()
    
    # SSD body
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd300_body(x)
    
    # Add multibox head for classification and regression
    num_priors = [12, 12, 12, 12, 12, 12]
    normalizations = [20, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes
    
    # parameters for prior boxes
    num_maps = len(source_layers)
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.aspect_ratios = [[1, 2, 3, 5, 7, 10] * 2] * num_maps
    #model.shifts = [[(0.0, 0.0)] * 6 + [(0.0, 1.0)] * 6] * num_maps
    model.shifts = [[(0.0, -0.5)] * 6 + [(0.0, 0.5)] * 6] * num_maps
    #model.minmax_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    
    return model

