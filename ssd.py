"""Keras implementation of SSD."""

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

from ssd_layers import Normalize
from ssd_layers import PriorBox


def SSD300(input_shape, num_classes=21):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(input_tensor)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)
    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)
    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)
    # Block 4
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)
    # Block 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)
    # FC6
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)
    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(fc6)
    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(conv6_1)
    # Block 7
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_2 = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv7_2')(conv7_2)
    # Block 8
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')(conv8_1)
    # Last Pool
    pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)
    # Prediction from conv4_3
    conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4_3)
    num_priors = 3
    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    conv4_3_norm_mbox_loc = x
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    conv4_3_norm_mbox_loc_flat = flatten(conv4_3_norm_mbox_loc)
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(conv4_3_norm)
    conv4_3_norm_mbox_conf = x
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    conv4_3_norm_mbox_conf_flat = flatten(conv4_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2], 
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')
    conv4_3_norm_mbox_priorbox = priorbox(conv4_3_norm)
    # Prediction from fc7
    num_priors = 6
    fc7_mbox_loc = Conv2D(num_priors * 4, (3, 3), padding='same', name='fc7_mbox_loc')(fc7)
    flatten = Flatten(name='fc7_mbox_loc_flat')
    fc7_mbox_loc_flat = flatten(fc7_mbox_loc)
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    fc7_mbox_conf = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(fc7)
    flatten = Flatten(name='fc7_mbox_conf_flat')
    fc7_mbox_conf_flat = flatten(fc7_mbox_conf)
    priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], 
                        variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')
    fc7_mbox_priorbox = priorbox(fc7)
    # Prediction from conv6_2
    num_priors = 6
    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv6_2_mbox_loc = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    conv6_2_mbox_loc_flat = flatten(conv6_2_mbox_loc)
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(conv6_2)
    conv6_2_mbox_conf = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    conv6_2_mbox_conf_flat = flatten(conv6_2_mbox_conf)
    priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')
    conv6_2_mbox_priorbox = priorbox(conv6_2)
    # Prediction from conv7_2
    num_priors = 6
    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')(conv7_2)
    conv7_2_mbox_loc = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    conv7_2_mbox_loc_flat = flatten(conv7_2_mbox_loc)
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(conv7_2)
    conv7_2_mbox_conf = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    conv7_2_mbox_conf_flat = flatten(conv7_2_mbox_conf)
    priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')
    conv7_2_mbox_priorbox = priorbox(conv7_2)
    # Prediction from conv8_2
    num_priors = 6
    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv8_2_mbox_loc = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    conv8_2_mbox_loc_flat = flatten(conv8_2_mbox_loc)
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(conv8_2)
    conv8_2_mbox_conf = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    conv8_2_mbox_conf_flat = flatten(conv8_2_mbox_conf)
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')
    conv8_2_mbox_priorbox = priorbox(conv8_2)
    # Prediction from pool6
    num_priors = 6
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(pool6)
    pool6_mbox_loc_flat = x
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Dense(num_priors * num_classes, name=name)(pool6)
    pool6_mbox_conf_flat = x
    priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    pool6_reshaped = Reshape(target_shape, name='pool6_reshaped')(pool6)
    pool6_mbox_priorbox = priorbox(pool6_reshaped)
    # Gather all predictions
    mbox_loc = concatenate([conv4_3_norm_mbox_loc_flat,
                            fc7_mbox_loc_flat,
                            conv6_2_mbox_loc_flat,
                            conv7_2_mbox_loc_flat,
                            conv8_2_mbox_loc_flat,
                            pool6_mbox_loc_flat],
                           axis=1, name='mbox_loc')
    mbox_conf = concatenate([conv4_3_norm_mbox_conf_flat,
                             fc7_mbox_conf_flat,
                             conv6_2_mbox_conf_flat,
                             conv7_2_mbox_conf_flat,
                             conv8_2_mbox_conf_flat,
                             pool6_mbox_conf_flat],
                            axis=1, name='mbox_conf')
    mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox,
                                 fc7_mbox_priorbox,
                                 conv6_2_mbox_priorbox,
                                 conv7_2_mbox_priorbox,
                                 conv8_2_mbox_priorbox,
                                 pool6_mbox_priorbox],
                                axis=1, name='mbox_priorbox')
    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4), name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc, 
                               mbox_conf, 
                               mbox_priorbox], 
                              axis=2, name='predictions')
    model = Model(input_tensor, predictions)
    return model


def SSD512(input_shape, num_classes=21):
    """SSD512 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (512, 512, 3) or (3, 512, 512)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(input_tensor)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)
    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)
    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)
    # Block 4
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)
    # Block 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)
    # FC6
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)
    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(fc6)
    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(conv6_1)
    # Block 7
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_2 = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv7_2')(conv7_2)
    # Block 8
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')(conv8_1)
    # Block 9
    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv9_2')(conv9_1)
    # Block 10 
    conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv10_1')(conv9_2)
    conv10_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv10_2')(conv10_1)
    # Last Pool
    pool6 = GlobalAveragePooling2D(name='pool6')(conv10_2)
    # Prediction from conv4_3
    conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4_3)
    num_priors = 4
    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    conv4_3_norm_mbox_loc = x
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    conv4_3_norm_mbox_loc_flat = flatten(conv4_3_norm_mbox_loc)
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(conv4_3_norm)
    conv4_3_norm_mbox_conf = x
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    conv4_3_norm_mbox_conf_flat = flatten(conv4_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 35.84, max_size=76.8, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')
    conv4_3_norm_mbox_priorbox = priorbox(conv4_3_norm)
    # Prediction from fc7
    num_priors = 6
    fc7_mbox_loc = Conv2D(num_priors * 4, (3, 3), padding='same', name='fc7_mbox_loc')(fc7)
    flatten = Flatten(name='fc7_mbox_loc_flat')
    fc7_mbox_loc_flat = flatten(fc7_mbox_loc)
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    fc7_mbox_conf = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(fc7)
    flatten = Flatten(name='fc7_mbox_conf_flat')
    fc7_mbox_conf_flat = flatten(fc7_mbox_conf)
    priorbox = PriorBox(img_size, 76.8, max_size=153.6, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')
    fc7_mbox_priorbox = priorbox(fc7)
    # Prediction from conv6_2
    num_priors = 6
    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv6_2_mbox_loc = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    conv6_2_mbox_loc_flat = flatten(conv6_2_mbox_loc)
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(conv6_2)
    conv6_2_mbox_conf = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    conv6_2_mbox_conf_flat = flatten(conv6_2_mbox_conf)
    priorbox = PriorBox(img_size, 153.6, max_size=230.4, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')
    conv6_2_mbox_priorbox = priorbox(conv6_2)
    # Prediction from conv7_2
    num_priors = 6
    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')(conv7_2)
    conv7_2_mbox_loc = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    conv7_2_mbox_loc_flat = flatten(conv7_2_mbox_loc)
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(conv7_2)
    conv7_2_mbox_conf = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    conv7_2_mbox_conf_flat = flatten(conv7_2_mbox_conf)
    priorbox = PriorBox(img_size, 230.4, max_size=307.2, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')
    conv7_2_mbox_priorbox = priorbox(conv7_2)
    # Prediction from conv8_2
    num_priors = 6
    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv8_2_mbox_loc = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    conv8_2_mbox_loc_flat = flatten(conv8_2_mbox_loc)
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(conv8_2)
    conv8_2_mbox_conf = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    conv8_2_mbox_conf_flat = flatten(conv8_2_mbox_conf)
    priorbox = PriorBox(img_size, 307.2, max_size=384.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')
    conv8_2_mbox_priorbox = priorbox(conv8_2)
    # Prediction from conv9_2
    num_priors = 4
    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv9_2_mbox_loc')(conv9_2)
    conv9_2_mbox_loc = x
    flatten = Flatten(name='conv9_2_mbox_loc_flat')
    conv9_2_mbox_loc_flat = flatten(conv9_2_mbox_loc)
    name = 'conv9_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(conv9_2)
    conv9_2_mbox_conf = x
    flatten = Flatten(name='conv9_2_mbox_conf_flat')
    conv9_2_mbox_conf_flat = flatten(conv9_2_mbox_conf)
    priorbox = PriorBox(img_size, 384.0, max_size=460.8, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv9_2_mbox_priorbox')
    conv9_2_mbox_priorbox = priorbox(conv9_2)
    # Prediction from pool6
    num_priors = 4
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(pool6)
    pool6_mbox_loc_flat = x
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Dense(num_priors * num_classes, name=name)(pool6)
    pool6_mbox_conf_flat = x
    priorbox = PriorBox(img_size, 460.8, max_size=537.6, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    pool6_reshaped = Reshape(target_shape, name='pool6_reshaped')(pool6)
    pool6_mbox_priorbox = priorbox(pool6_reshaped)
    # Gather all predictions
    mbox_loc = concatenate([conv4_3_norm_mbox_loc_flat,
                            fc7_mbox_loc_flat,
                            conv6_2_mbox_loc_flat,
                            conv7_2_mbox_loc_flat,
                            conv8_2_mbox_loc_flat,
                            conv9_2_mbox_loc_flat,
                            pool6_mbox_loc_flat],
                           axis=1, name='mbox_loc')
    mbox_conf = concatenate([conv4_3_norm_mbox_conf_flat,
                             fc7_mbox_conf_flat,
                             conv6_2_mbox_conf_flat,
                             conv7_2_mbox_conf_flat,
                             conv8_2_mbox_conf_flat,
                             conv9_2_mbox_conf_flat,
                             pool6_mbox_conf_flat],
                            axis=1, name='mbox_conf')
    mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox,
                                 fc7_mbox_priorbox,
                                 conv6_2_mbox_priorbox,
                                 conv7_2_mbox_priorbox,
                                 conv8_2_mbox_priorbox,
                                 conv9_2_mbox_priorbox,
                                 pool6_mbox_priorbox],
                                axis=1, name='mbox_priorbox')
    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4), name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc,
                               mbox_conf,
                               mbox_priorbox],
                              axis=2, name='predictions')
    model = Model(input_tensor, predictions)
    return model
