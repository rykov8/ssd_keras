"""Keras implementation of TextBoxes++."""

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
from ssd_model import ssd512_body
from ssd_model_dense import dsod512_body


def multibox_head(source_layers, num_priors, normalizations=None, softmax=True):
    
    num_classes = 2
    class_activation = 'softmax' if softmax else 'sigmoid'

    mbox_conf = []
    mbox_loc = []
    mbox_quad = []
    mbox_rbox = []
    for i in range(len(source_layers)):
        x = source_layers[i]
        name = x.name.split('/')[0]
        
        # normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + '_norm'
            x = Normalize(normalizations[i], name=name)(x)
            
        # confidence
        name1 = name + '_mbox_conf'
        x1 = Conv2D(num_priors[i] * num_classes, (3, 5), padding='same', name=name1)(x)
        x1 = Flatten(name=name1+'_flat')(x1)
        mbox_conf.append(x1)

        # location, Delta(x,y,w,h)
        name2 = name + '_mbox_loc'
        x2 = Conv2D(num_priors[i] * 4, (3, 5), padding='same', name=name2)(x)
        x2 = Flatten(name=name2+'_flat')(x2)
        mbox_loc.append(x2)
        
        # quadrilateral, Delta(x1,y1,x2,y2,x3,y3,x4,y4)
        name3 = name + '_mbox_quad'
        x3 = Conv2D(num_priors[i] * 8, (3, 5), padding='same', name=name3)(x)
        x3 = Flatten(name=name3+'_flat')(x3)
        mbox_quad.append(x3)

        # rotated rectangle, Delta(x1,y1,x2,y2,h)
        name4 = name + '_mbox_rbox'
        x4 = Conv2D(num_priors[i] * 5, (3, 5), padding='same', name=name4)(x)
        x4 = Flatten(name=name4+'_flat')(x4)
        mbox_rbox.append(x4)
        
    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation(class_activation, name='mbox_conf_final')(mbox_conf)
    
    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)
    
    mbox_quad = concatenate(mbox_quad, axis=1, name='mbox_quad')
    mbox_quad = Reshape((-1, 8), name='mbox_quad_final')(mbox_quad)
    
    mbox_rbox = concatenate(mbox_rbox, axis=1, name='mbox_rbox')
    mbox_rbox = Reshape((-1, 5), name='mbox_rbox_final')(mbox_rbox)

    predictions = concatenate([mbox_loc, mbox_quad, mbox_rbox, mbox_conf], axis=2, name='predictions')
    
    return predictions


def TBPP512(input_shape=(512, 512, 3), softmax=True):
    """TextBoxes++512 architecture.

    # Arguments
        input_shape: Shape of the input image.
    
    # References
        - [TextBoxes++: A Single-Shot Oriented Scene Text Detector](https://arxiv.org/abs/1801.02765)
    """
    
    # SSD body
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_body(x)
    
    num_maps = len(source_layers)
    
    # Add multibox head for classification and regression
    num_priors = [14] * num_maps
    normalizations = [1] * num_maps
    output_tensor = multibox_head(source_layers, num_priors, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    
    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    
    model.aspect_ratios = [[1,2,3,5,1/2,1/3,1/5] * 2] * num_maps
    #model.shifts = [[(0.0, 0.0)] * 7 + [(0.0, 1.0)] * 7] * num_maps
    model.shifts = [[(0.0, -0.5)] * 7 + [(0.0, 0.5)] * 7] * num_maps
    model.special_ssd_boxes = False
    model.scale = 0.5
    
    return model

def TBPP512_dense(input_shape=(512, 512, 3), softmax=True):
    """DenseNet based Architecture for TextBoxes++512.
    """
    
    # DSOD body
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod512_body(x)
    
    num_maps = len(source_layers)
    
    # Add multibox head for classification and regression
    num_priors = [14] * num_maps
    normalizations = [1] * num_maps
    output_tensor = multibox_head(source_layers, num_priors, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    
    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    
    model.aspect_ratios = [[1,2,3,5,1/2,1/3,1/5] * 2] * num_maps
    #model.shifts = [[(0.0, 0.0)] * 7 + [(0.0, 1.0)] * 7] * num_maps
    model.shifts = [[(0.0, -0.5)] * 7 + [(0.0, 0.5)] * 7] * num_maps
    model.special_ssd_boxes = False
    model.scale = 0.5
    
    return model

