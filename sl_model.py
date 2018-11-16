"""Keras implementation of SegLink."""

from keras.models import Model
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Reshape

from utils.layers import Normalize
from ssd_model import ssd512_body
from ssd_model_dense import dsod512_body, ssd384x512_dense_body
from ssd_model_resnet import ssd512_resnet_body


def multibox_head(source_layers, num_priors, normalizations=None, softmax=True):
    
    num_classes = 2
    class_activation = 'softmax' if softmax else 'sigmoid'
    
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
            
        # confidence
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
    mbox_conf = Activation(class_activation, name='mbox_conf_final')(mbox_conf)
    
    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 5), name='mbox_loc_final')(mbox_loc)

    #link_interlayer_conf = concatenate(link_interlayer_conf, axis=1, name='link_interlayer_conf')
    #link_interlayer_conf = Reshape((-1, num_classes * 8), name='link_interlayer_conf_logits')(link_interlayer_conf)
    #link_interlayer_conf = Activation(class_activation, name='link_interlayer_conf_final')(link_interlayer_conf)
    
    #link_crosslayer_conf = concatenate(link_crosslayer_conf, axis=1, name='link_crosslayer_conf')
    #link_crosslayer_conf = Reshape((-1, num_classes * 4), name='link_crosslayer_conf_logits')(link_crosslayer_conf)
    #link_crosslayer_conf = Activation(class_activation, name='link_crosslayer_conf_final')(link_crosslayer_conf)
    
    link_interlayer_conf = concatenate(link_interlayer_conf, axis=1, name='link_interlayer_conf')
    link_interlayer_conf = Reshape((-1, num_classes), name='link_interlayer_conf_logits')(link_interlayer_conf)
    link_interlayer_conf = Activation(class_activation, name='link_interlayer_conf_softmax')(link_interlayer_conf)
    link_interlayer_conf = Reshape((-1, num_classes * 8), name='link_interlayer_conf_final')(link_interlayer_conf)
    
    link_crosslayer_conf = concatenate(link_crosslayer_conf, axis=1, name='link_crosslayer_conf')
    link_crosslayer_conf = Reshape((-1, num_classes), name='link_crosslayer_conf_logits')(link_crosslayer_conf)
    link_crosslayer_conf = Activation(class_activation, name='link_crosslayer_conf_softmax')(link_crosslayer_conf)
    link_crosslayer_conf = Reshape((-1, num_classes * 4), name='link_crosslayer_conf_final')(link_crosslayer_conf)
    
    predictions = concatenate([
            mbox_conf, 
            mbox_loc,
            link_interlayer_conf, 
            link_crosslayer_conf
            ], axis=2, name='predictions')
    
    return predictions


def SL512(input_shape=(512, 512, 3), softmax=True):
    """SegLink512 architecture.

    # Arguments
        input_shape: Shape of the input image.

    # References
        https://arxiv.org/abs/1703.06520
    """
    
    # SSD body
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_body(x)
    
    # Add multibox head for classification and regression
    num_priors = [1, 1, 1, 1, 1, 1, 1]
    normalizations = [20, -1, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(source_layers, num_priors, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    
    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    
    return model


def DSODSL512(input_shape=(512, 512, 3), activation='relu', softmax=True):
    """DenseNet based Architecture for SegLink512.
    
    # Arguments
        input_shape: Shape of the input image.

    # References
        https://arxiv.org/abs/1708.01241
    """
    
    # DSOD body
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod512_body(x, activation=activation)
    
    # Add multibox head for classification and regression
    num_priors = [1, 1, 1, 1, 1, 1, 1]
    normalizations = [20, -1, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(source_layers, num_priors, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    
    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    
    return model


def SL384x512_dense(input_shape=(384,512,3), activation='relu'):
    
    # body
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd384x512_dense_body(x, activation=activation)
    
    # Add multibox head for classification and regression
    num_priors = [1, 1, 1, 1, 1, 1]
    normalizations = [20, 20, 20, 20, 20, 20]
    output_tensor = multibox_head(source_layers, num_priors, normalizations)
    model = Model(input_tensor, output_tensor)
    
    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    
    return model


def SL512_resnet(input_shape=(512, 512, 3), activation='relu', softmax=True):
    
    # TODO: it does not converge!
    
    # DSOD body
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_resnet_body(x, activation=activation)
    
    # Add multibox head for classification and regression
    num_priors = [1, 1, 1, 1, 1, 1, 1]
    normalizations = [20, 20, 20, 20, 20, 20, 20]
    output_tensor = multibox_head(source_layers, num_priors, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    
    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    
    return model
