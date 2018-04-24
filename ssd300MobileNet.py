"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D,SeparableConv2D
from keras.layers import Dropout,BatchNormalization
from keras.layers import AlphaDropout,GaussianDropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers import Reshape
from keras.models import Model
from ssd_layers import PriorBox
from keras.models import Sequential
from keras.applications import MobileNet




def SSD(input_shape, num_classes):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    img_size=(input_shape[1],input_shape[0])
    input_shape=(input_shape[1],input_shape[0],3)
    mobilenet_input_shape=(224,224,3)
    net={}
    net['input']=Input(input_shape)
    mobilenet=MobileNet(input_shape=mobilenet_input_shape,include_top=False,weights='imagenet')
    FeatureExtractor=Model(inputs=mobilenet.input,outputs=mobilenet.get_layer('conv_dw_11_relu').output)
    conv11=FeatureExtractor(net['input'])

    net['conv11'] = Conv2D(512, (1, 1),  padding='same', name='conv11')(conv11)
    net['conv11'] = BatchNormalization( momentum=0.99, name='bn11')(net['conv11'])
    net['conv11'] = Activation('relu')(net['conv11'])
    # Block
    #(19,19)
    net['conv12dw'] = SeparableConv2D(512, (3, 3),strides=(2, 2),  padding='same', name='conv12dw')(net['conv11'])
    net['conv12dw'] = BatchNormalization( momentum=0.99, name='bn12dw')(net['conv12dw'])
    net['conv12dw'] = Activation('relu')(net['conv12dw'])
    net['conv12'] = Conv2D(1024, (1, 1), padding='same',name='conv12')(net['conv12dw'])
    net['conv12'] = BatchNormalization( momentum=0.99, name='bn12')(net['conv12'])
    net['conv12'] = Activation('relu')(net['conv12'])
    net['conv13dw'] = SeparableConv2D(1024, (3, 3), padding='same',name='conv13dw')(net['conv12'])
    net['conv13dw'] = BatchNormalization( momentum=0.99, name='bn13dw')(net['conv13dw'])
    net['conv13dw'] = Activation('relu')(net['conv13dw'])
    net['conv13'] = Conv2D(1024, (1, 1), padding='same',name='conv13')(net['conv13dw'])
    net['conv13'] = BatchNormalization( momentum=0.99, name='bn13')(net['conv13'])
    net['conv13'] = Activation('relu')(net['conv13'])
    net['conv14_1'] = Conv2D(256, (1, 1),  padding='same', name='conv14_1')(net['conv13'])
    net['conv14_1'] = BatchNormalization( momentum=0.99, name='bn14_1')(net['conv14_1'])
    net['conv14_1'] = Activation('relu')(net['conv14_1'])
    net['conv14_2'] = Conv2D(512, (3, 3), strides=(2, 2),  padding='same', name='conv14_2')(net['conv14_1'])
    net['conv14_2'] = BatchNormalization( momentum=0.99, name='bn14_2')(net['conv14_2'])
    net['conv14_2'] = Activation('relu')(net['conv14_2'])
    net['conv15_1'] = Conv2D(128, (1, 1), padding='same',name='conv15_1')(net['conv14_2'])
    net['conv15_1'] = BatchNormalization( momentum=0.99, name='bn15_1')(net['conv15_1'])
    net['conv15_1'] = Activation('relu')(net['conv15_1'])
    net['conv15_2'] = Conv2D(256, (3, 3), strides=(2, 2), padding='same',name='conv15_2')(net['conv15_1'])
    net['conv15_2'] = BatchNormalization( momentum=0.99, name='bn15_2')(net['conv15_2'])
    net['conv15_2'] = Activation('relu')(net['conv15_2'])
    net['conv16_1'] = Conv2D(128, (1, 1),  padding='same', name='conv16_1')(net['conv15_2'])
    net['conv16_1'] = BatchNormalization( momentum=0.99, name='bn16_1')(net['conv16_1'])
    net['conv16_1'] = Activation('relu')(net['conv16_1'])
    net['conv16_2'] = Conv2D(256, (3, 3), strides=(2, 2),  padding='same', name='conv16_2')(net['conv16_1'])
    net['conv16_2'] = BatchNormalization( momentum=0.99, name='bn16_2')(net['conv16_2'])
    net['conv16_2'] = Activation('relu')(net['conv16_2'])
    net['conv17_1'] = Conv2D(64, (1, 1),  padding='same', name='conv17_1')(net['conv16_2'])
    net['conv17_1'] = BatchNormalization( momentum=0.99, name='bn17_1')(net['conv17_1'])
    net['conv17_1'] = Activation('relu')(net['conv17_1'])
    net['conv17_2'] = Conv2D(128, (3, 3), strides=(2, 2),  padding='same', name='conv17_2')(net['conv17_1'])
    net['conv17_2'] = BatchNormalization( momentum=0.99, name='bn17_2')(net['conv17_2'])
    net['conv17_2'] = Activation('relu')(net['conv17_2'])

    #Prediction from conv11
    num_priors = 3
    x = Conv2D(num_priors * 4, (1,1), padding='same',name='conv11_mbox_loc')(net['conv11'])
    net['conv11_mbox_loc'] = x
    flatten = Flatten(name='conv11_mbox_loc_flat')
    net['conv11_mbox_loc_flat'] = flatten(net['conv11_mbox_loc'])
    name = 'conv11_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (1,1), padding='same',name=name)(net['conv11'])
    net['conv11_mbox_conf'] = x
    flatten = Flatten(name='conv11_mbox_conf_flat')
    net['conv11_mbox_conf_flat'] = flatten(net['conv11_mbox_conf'])
    priorbox = PriorBox(img_size,60,max_size=None, aspect_ratios=[2],variances=[0.1, 0.1, 0.2, 0.2],name='conv11_mbox_priorbox')
    net['conv11_mbox_priorbox'] = priorbox(net['conv11'])
    # Prediction from conv13
    num_priors = 6
    net['conv13_mbox_loc'] = Conv2D(num_priors * 4, (1,1),padding='same',name='conv13_mbox_loc')(net['conv13'])
    flatten = Flatten(name='conv13_mbox_loc_flat')
    net['conv13_mbox_loc_flat'] = flatten(net['conv13_mbox_loc'])
    name = 'conv13_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['conv13_mbox_conf'] = Conv2D(num_priors * num_classes, (1,1),padding='same',name=name)(net['conv13'])
    flatten = Flatten(name='conv13_mbox_conf_flat')
    net['conv13_mbox_conf_flat'] = flatten(net['conv13_mbox_conf'])
    priorbox = PriorBox(img_size, 105.0, max_size=150.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv13_mbox_priorbox')
    net['conv13_mbox_priorbox'] = priorbox(net['conv13'])
    # Prediction from conv12
    num_priors = 6
    x = Conv2D(num_priors * 4, (1,1), padding='same',name='conv14_2_mbox_loc')(net['conv14_2'])
    net['conv14_2_mbox_loc'] = x
    flatten = Flatten(name='conv14_2_mbox_loc_flat')
    net['conv14_2_mbox_loc_flat'] = flatten(net['conv14_2_mbox_loc'])
    name = 'conv14_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (1,1), padding='same',name=name)(net['conv14_2'])
    net['conv14_2_mbox_conf'] = x
    flatten = Flatten(name='conv14_2_mbox_conf_flat')
    net['conv14_2_mbox_conf_flat'] = flatten(net['conv14_2_mbox_conf'])
    priorbox = PriorBox(img_size, 150, max_size=195.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv14_2_mbox_priorbox')
    net['conv14_2_mbox_priorbox'] = priorbox(net['conv14_2'])
    # Prediction from conv15_2_mbox
    num_priors = 6
    x = Conv2D(num_priors * 4, (1,1), padding='same',name='conv15_2_mbox_loc')(net['conv15_2'])
    net['conv15_2_mbox_loc'] = x
    flatten = Flatten(name='conv15_2_mbox_loc_flat')
    net['conv15_2_mbox_loc_flat'] = flatten(net['conv15_2_mbox_loc'])
    name = 'conv15_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (1,1), padding='same',name=name)(net['conv15_2'])
    net['conv15_2_mbox_conf'] = x
    flatten = Flatten(name='conv15_2_mbox_conf_flat')
    net['conv15_2_mbox_conf_flat'] = flatten(net['conv15_2_mbox_conf'])
    priorbox = PriorBox(img_size, 195.0, max_size=240.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv15_2_mbox_priorbox')
    net['conv15_2_mbox_priorbox'] = priorbox(net['conv15_2'])

    # Prediction from conv16_2
    num_priors = 6
    x = Conv2D(num_priors * 4, (1,1), padding='same',name='conv16_2_mbox_loc')(net['conv16_2'])
    net['conv16_2_mbox_loc'] = x
    flatten = Flatten(name='conv16_2_mbox_loc_flat')
    net['conv16_2_mbox_loc_flat'] = flatten(net['conv16_2_mbox_loc'])
    name = 'conv16_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (1,1), padding='same',name=name)(net['conv16_2'])
    net['conv16_2_mbox_conf'] = x
    flatten = Flatten(name='conv16_2_mbox_conf_flat')
    net['conv16_2_mbox_conf_flat'] = flatten(net['conv16_2_mbox_conf'])
    priorbox = PriorBox(img_size, 240.0, max_size=285.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv16_2_mbox_priorbox')
    net['conv16_2_mbox_priorbox'] = priorbox(net['conv16_2'])

    # Prediction from conv17_2
    num_priors = 6
    x = Conv2D(num_priors * 4,(1, 1), padding='same', name='conv17_2_mbox_loc')(net['conv17_2'])
    net['conv17_2_mbox_loc'] = x
    flatten = Flatten(name='conv17_2_mbox_loc_flat')
    net['conv17_2_mbox_loc_flat'] = flatten(net['conv17_2_mbox_loc'])
    name = 'conv17_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (1,1), padding='same', name=name)(net['conv17_2'])
    net['conv17_2_mbox_conf'] = x
    flatten = Flatten(name='conv17_2_mbox_conf_flat')
    net['conv17_2_mbox_conf_flat'] = flatten(net['conv17_2_mbox_conf'])
    priorbox = PriorBox(img_size, 285.0, max_size=300.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2],name='conv17_2_mbox_priorbox')
    net['conv17_2_mbox_priorbox'] = priorbox(net['conv17_2'])

    # Gather all predictions
    net['mbox_loc'] = concatenate([net['conv11_mbox_loc_flat'],net['conv13_mbox_loc_flat'],net['conv14_2_mbox_loc_flat'],net['conv15_2_mbox_loc_flat'],net['conv16_2_mbox_loc_flat'],net['conv17_2_mbox_loc_flat']],axis=1, name='mbox_loc')
    net['mbox_conf'] = concatenate([net['conv11_mbox_conf_flat'],net['conv13_mbox_conf_flat'],net['conv14_2_mbox_conf_flat'],net['conv15_2_mbox_conf_flat'],net['conv16_2_mbox_conf_flat'],net['conv17_2_mbox_conf_flat']],axis=1, name='mbox_conf')
    net['mbox_priorbox'] = concatenate([net['conv11_mbox_priorbox'],net['conv13_mbox_priorbox'],net['conv14_2_mbox_priorbox'],net['conv15_2_mbox_priorbox'],net['conv16_2_mbox_priorbox'],net['conv17_2_mbox_priorbox']],axis=1,name='mbox_priorbox')
    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4
    net['mbox_loc'] = Reshape((num_boxes, 4),name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((num_boxes, num_classes),name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax',name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = concatenate([net['mbox_loc'],net['mbox_conf'],net['mbox_priorbox']],axis=2,name='predictions')
    model = Model(net['input'], net['predictions'])
    return model
