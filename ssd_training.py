"""SSD training utils."""

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import json
import time

from keras.callbacks import Callback
from math import log10, floor, ceil


def smooth_l1_loss(y_true, y_pred):
    """Compute L1-smooth loss.

    # Arguments
        y_true: Ground truth bounding boxes,
            tensor of shape (?, num_boxes, 4).
        y_pred: Predicted bounding boxes,
            tensor of shape (?, num_boxes, 4).

    # Returns
        l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).

    # References
        https://arxiv.org/abs/1504.08083
    """
    abs_loss = tf.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    return tf.reduce_sum(l1_loss, -1)

def softmax_loss(y_true, y_pred):
    """Compute cross entropy loss aka softmax loss.

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).

    # Returns
        softmax_loss: Softmax loss, tensor of shape (?, num_boxes).
    """
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1. - eps)
    softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
    return softmax_loss


class SSDLoss(object):
    """Multibox loss for SSD.
    
    # Arguments
        alpha: Weight of L1-smooth loss.
        neg_pos_ratio: Max ratio of negative to positive boxes in loss.
        negatives_for_hard: Number of negative boxes to consider
            if there is no positive boxes in batch.
        
    # References
        https://arxiv.org/abs/1512.02325
    """

    def __init__(self, alpha=1.0, neg_pos_ratio=3.0, negatives_for_hard=100.0):
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.metrics = []
    
    def compute(self, y_true, y_pred):
        # y.shape (batches, priors, 4 x segment_offset + n x class_label)
        # TODO: negatives_for_hard?
        #       mask based on y_true or y_pred?
        
        batch_size = tf.shape(y_true)[0]
        num_priors = tf.shape(y_true)[1]
        num_classes = tf.shape(y_true)[2] - 4
        eps = K.epsilon()
        
        # confidence loss
        conf_true = tf.reshape(y_true[:,:,4:], [-1, num_classes])
        conf_pred = tf.reshape(y_pred[:,:,4:], [-1, num_classes])
        
        conf_loss = softmax_loss(conf_true, conf_pred)
        class_true = tf.argmax(conf_true, axis=1)
        class_pred = tf.argmax(conf_pred, axis=1)
        
        neg_mask_float = conf_true[:,0]
        neg_mask = tf.cast(neg_mask_float, tf.bool)
        pos_mask = tf.logical_not(neg_mask)
        pos_mask_float = tf.cast(pos_mask, tf.float32)
        num_total = tf.cast(tf.shape(conf_true)[0], tf.float32)
        num_pos = tf.reduce_sum(pos_mask_float)
        num_neg = num_total - num_pos
        
        pos_conf_loss = tf.reduce_sum(conf_loss * pos_mask_float)
        
        ## take only false positives for hard negative mining
        #false_pos_mask = tf.logical_and(neg_mask, tf.not_equal(class_pred, 0))
        #num_false_pos = tf.reduce_sum(tf.cast(false_pos_mask, tf.float32))
        #num_neg = tf.minimum(self.neg_pos_ratio * num_pos, num_false_pos)
        #neg_conf_loss = tf.boolean_mask(conf_loss, false_pos_mask)
        
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos, num_neg)
        neg_conf_loss = tf.boolean_mask(conf_loss, neg_mask)
        
        vals, idxs = tf.nn.top_k(neg_conf_loss, k=tf.cast(num_neg, tf.int32))
        #neg_conf_loss = tf.reduce_sum(tf.gather(neg_conf_loss, idxs))
        neg_conf_loss = tf.reduce_sum(vals)
        
        conf_loss = (pos_conf_loss + neg_conf_loss) / (num_pos + num_neg + eps)
        
        # offset loss
        loc_true = tf.reshape(y_true[:,:,0:4], [-1, 4])
        loc_pred = tf.reshape(y_pred[:,:,0:4], [-1, 4])
        
        loc_loss = smooth_l1_loss(loc_true, loc_pred)
        pos_loc_loss = tf.reduce_sum(loc_loss * pos_mask_float) # only for positives
        
        loc_loss = pos_loc_loss / (num_pos + eps)
        
        # total loss
        total_loss = conf_loss + self.alpha * loc_loss
        
        # metrics
        pos_conf_loss = pos_conf_loss / (num_pos + eps)
        neg_conf_loss = neg_conf_loss / (num_neg + eps)
        pos_loc_loss = pos_loc_loss / (num_pos + eps)
        
        precision, recall, accuracy, fmeasure = compute_metrics(class_true, class_pred, conf_loss)
        # TODO: use conf, not conf_loss
        
        def make_fcn(t):
            return lambda y_true, y_pred: t
        for name in ['num_pos', 
                     'num_neg', 
                     'pos_conf_loss', 
                     'neg_conf_loss', 
                     'pos_loc_loss', 
                     'precision', 
                     'recall',
                     'accuracy',
                     'fmeasure', 
                    ]:
            f = make_fcn(eval(name))
            f.__name__ = name
            self.metrics.append(f)
        
        return total_loss

