"""Keras implementation of SegLink."""

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

from ssd_model import ssd512_body
from dsod_model import dsod512_body
from ssd_layers import Normalize


def multibox_head(source_layers, num_priors, num_classes, normalizations=None):
    
    mbox_conf = []
    mbox_loc = []
    link_interlayer_conf = []
    link_crosslayer_conf = []
    for i in range(len(source_layers)):
        x = source_layers[i]
        name = x.name.split('/')[0]
        
        # normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + '_norm'
            x = Normalize(normalizations[i], name=name)(x)
            
        # confidenc
        name1 = name + '_mbox_conf'
        x1 = Conv2D(num_priors[i] * num_classes, 3, padding='same', name=name1)(x)
        x1 = Flatten(name=name1+'_flat')(x1)
        mbox_conf.append(x1)

        # location
        name2 = name + '_mbox_loc'
        x2 = Conv2D(num_priors[i] * 5, 3, padding='same', name=name2)(x)
        x2 = Flatten(name=name2+'_flat')(x2)
        mbox_loc.append(x2)
        
        # link interlayer confidenc
        name3 = name + '_link_interlayer_conf'
        x3 = Conv2D(num_priors[i] * num_classes * 8, 3, padding='same', name=name3)(x)
        x3 = Flatten(name=name3+'_flat')(x3)
        link_interlayer_conf.append(x3)
        
        # link crosslayer confidenc
        name4 = name + '_link_crosslayer_conf'
        x4 = Conv2D(num_priors[i] * num_classes * 4, 3, padding='same', name=name4)(x)
        x4 = Flatten(name=name4+'_flat')(x4)
        link_crosslayer_conf.append(x4)

    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)
    
    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 5), name='mbox_loc_final')(mbox_loc)

    #link_interlayer_conf = concatenate(link_interlayer_conf, axis=1, name='link_interlayer_conf')
    #link_interlayer_conf = Reshape((-1, num_classes * 8), name='link_interlayer_conf_logits')(link_interlayer_conf)
    #link_interlayer_conf = Activation('softmax', name='link_interlayer_conf_final')(link_interlayer_conf)
    
    #link_crosslayer_conf = concatenate(link_crosslayer_conf, axis=1, name='link_crosslayer_conf')
    #link_crosslayer_conf = Reshape((-1, num_classes * 4), name='link_crosslayer_conf_logits')(link_crosslayer_conf)
    #link_crosslayer_conf = Activation('softmax', name='link_crosslayer_conf_final')(link_crosslayer_conf)
    
    link_interlayer_conf = concatenate(link_interlayer_conf, axis=1, name='link_interlayer_conf')
    link_interlayer_conf = Reshape((-1, num_classes), name='link_interlayer_conf_logits')(link_interlayer_conf)
    link_interlayer_conf = Activation('softmax', name='link_interlayer_conf_softmax')(link_interlayer_conf)
    link_interlayer_conf = Reshape((-1, num_classes * 8), name='link_interlayer_conf_final')(link_interlayer_conf)
    
    link_crosslayer_conf = concatenate(link_crosslayer_conf, axis=1, name='link_crosslayer_conf')
    link_crosslayer_conf = Reshape((-1, num_classes), name='link_crosslayer_conf_logits')(link_crosslayer_conf)
    link_crosslayer_conf = Activation('softmax', name='link_crosslayer_conf_softmax')(link_crosslayer_conf)
    link_crosslayer_conf = Reshape((-1, num_classes * 4), name='link_crosslayer_conf_final')(link_crosslayer_conf)
    
    predictions = concatenate([
            mbox_conf, 
            mbox_loc,
            link_interlayer_conf, 
            link_crosslayer_conf
            ], axis=2, name='predictions')
    
    return predictions


def SL512(input_shape=(512, 512, 3), num_classes=2):
    """SegLink512 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1703.06520
    """
    K.clear_session()
    
    # SSD body
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_body(x)
    
    # Add multibox head for classification and regression
    num_priors = [1, 1, 1, 1, 1, 1, 1]
    normalizations = [20, -1, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations)
    model = Model(input_tensor, output_tensor)

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.source_layers_names = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2', 'conv10_2']
    
    return model

