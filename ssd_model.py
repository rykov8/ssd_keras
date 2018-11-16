"""Keras implementation of SSD."""

from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

from utils.layers import Normalize
from ssd_model_dense import dsod300_body, dsod512_body
from ssd_model_resnet import ssd512_resnet_body


def ssd300_body(x):
    
    source_layers = []
    
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(512, 3, strides=2, padding='valid', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='valid', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='valid', name='conv9_2', activation='relu')(x)
    source_layers.append(x)
    
    return source_layers


def ssd512_body(x):
    
    source_layers = []
    
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(512, 3, strides=2, padding='valid', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv9_2', activation='relu')(x)
    source_layers.append(x)
    # Block 10 
    x = Conv2D(128, 1, strides=1, padding='same', name='conv10_1', activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(256, 4, strides=2, padding='valid', name='conv10_2', activation='relu')(x)
    source_layers.append(x)
    
    return source_layers


def multibox_head(source_layers, num_priors, num_classes, normalizations=None, softmax=True):

    class_activation = 'softmax' if softmax else 'sigmoid'

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
        x1 = Conv2D(num_priors[i] * num_classes, 3, padding='same', name=name1)(x)
        x1 = Flatten(name=name1+'_flat')(x1)
        mbox_conf.append(x1)

        # location
        name2 = name + '_mbox_loc'
        x2 = Conv2D(num_priors[i] * 4, 3, padding='same', name=name2)(x)
        x2 = Flatten(name=name2+'_flat')(x2)
        mbox_loc.append(x2)

    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation(class_activation, name='mbox_conf_final')(mbox_conf)
    
    predictions = concatenate([mbox_loc, mbox_conf], axis=2, name='predictions')
    
    return predictions


def SSD300(input_shape=(300, 300, 3), num_classes=21, softmax=True):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.
    
    # Notes
        In order to stay compatible with pre-trained models, the parameters 
        were chosen as in the caffee implementation.
    
    # References
        https://arxiv.org/abs/1512.02325
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd300_body(x)
    
    # Add multibox head for classification and regression
    num_priors = [4, 6, 6, 6, 4, 4]
    normalizations = [20, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    # stay compatible with caffe models
    model.aspect_ratios = [[1,2,1/2], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,1/2], [1,2,1/2]]
    model.minmax_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
    model.steps = [8, 16, 32, 64, 100, 300]
    model.special_ssd_boxes = True
    
    return model


def SSD512(input_shape=(512, 512, 3), num_classes=21, softmax=True):
    """SSD512 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.
    
    # Notes
        In order to stay compatible with pre-trained models, the parameters 
        were chosen as in the caffee implementation.
    
    # References
        https://arxiv.org/abs/1512.02325
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_body(x)
    
    # Add multibox head for classification and regression
    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, -1, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    # stay compatible with caffe models
    model.aspect_ratios = [[1,2,1/2], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,3,1/2,1/3], [1,2,1/2], [1,2,1/2]]
    #model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460), (460, 537)]
    model.minmax_sizes = [(20.48, 51.2), (51.2, 133.12), (133.12, 215.04), (215.04, 296.96), (296.96, 378.88), (378.88, 460.8), (460.8, 542.72)]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    model.special_ssd_boxes = True
    
    return model


def DSOD300(input_shape=(300, 300, 3), num_classes=21, activation='relu', softmax=True):
    """DSOD, DenseNet based SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.
        activation: Type of activation functions.
    
    # References
        https://arxiv.org/abs/1708.01241
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod300_body(x, activation=activation)

    num_priors = [4, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20]

    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.aspect_ratios = [[1,2,1/2], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,1/2], [1,2,1/2]]
    model.minmax_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
    model.steps = [8, 16, 32, 64, 100, 300]
    model.special_ssd_boxes = True
    
    return model

SSD300_dense = DSOD300


def DSOD512(input_shape=(512, 512, 3), num_classes=21, activation='relu', softmax=True):
    """DSOD, DenseNet based SSD512 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.
        activation: Type of activation functions.
    
    # References
        https://arxiv.org/abs/1708.01241
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod512_body(x, activation=activation)

    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20, 20]

    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.aspect_ratios = [[1,2,1/2], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,3,1/2,1/3], [1,2,1/2], [1,2,1/2]]
    model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460), (460, 537)]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    model.special_ssd_boxes = True
    
    return model

SSD512_dense = DSOD512


def SSD512_resnet(input_shape=(512, 512, 3), num_classes=21, softmax=True):
    
    # TODO: it does not converge!
    
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_resnet_body(x)
    
    # Add multibox head for classification and regression
    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20, 20]
    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    # stay compatible with caffe models
    model.aspect_ratios = [[1,2,1/2], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,1/2,3,1/3], [1,2,3,1/2,1/3], [1,2,1/2], [1,2,1/2]]
    model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460), (460, 537)]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    model.special_ssd_boxes = True
    
    return model
