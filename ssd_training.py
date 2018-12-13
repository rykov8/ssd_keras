"""SSD training utils."""

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

from utils.training import smooth_l1_loss, softmax_loss, focal_loss
from utils.training import plot_log


def compute_metrics(class_true, class_pred, conf, top_k=100):
    """Compute precision, recall, accuracy and f-measure for top_k predictions.
    
    from top_k predictions that are TP FN or FP (TN kept out)
    """

    top_k = tf.cast(top_k, tf.int32)
    eps = K.epsilon()
    
    mask = tf.greater(class_true + class_pred, 0)
    #mask = tf.logical_or(tf.greater(class_true, 0), tf.greater(class_pred, 0))
    mask_float = tf.cast(mask, tf.float32)
    
    vals, idxs = tf.nn.top_k(conf * mask_float, k=top_k)
    
    top_k_class_true = tf.gather(class_true, idxs)
    top_k_class_pred = tf.gather(class_pred, idxs)
    
    true_mask = tf.equal(top_k_class_true, top_k_class_pred)
    false_mask = tf.logical_not(true_mask)
    pos_mask = tf.greater(top_k_class_pred, 0)
    neg_mask = tf.logical_not(pos_mask)
    
    tp = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pos_mask), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(false_mask, pos_mask), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(false_mask, neg_mask), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, neg_mask), tf.float32))
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    fmeasure = 2 * (precision * recall) / (precision + recall + eps)
    
    return precision, recall, accuracy, fmeasure


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
        conf = tf.reduce_max(conf_pred, axis=1)
        
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
        
        precision, recall, accuracy, fmeasure = compute_metrics(class_true, class_pred, conf, top_k=100*batch_size)
        
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


class SSDFocalLoss(object):

    def __init__(self, lambda_conf=10000.0, lambda_offsets=1.0, class_weights=1.0):
        self.lambda_conf = lambda_conf
        self.lambda_offsets = lambda_offsets
        self.class_weights = class_weights
        self.metrics = []
    
    def compute(self, y_true, y_pred):
        # y.shape (batches, priors, 4 x bbox_offset + n x class_label)
        
        batch_size = tf.shape(y_true)[0]
        num_priors = tf.shape(y_true)[1]
        num_classes = tf.shape(y_true)[2] - 4
        eps = K.epsilon()
        
        # confidence loss
        conf_true = tf.reshape(y_true[:,:,4:], [-1, num_classes])
        conf_pred = tf.reshape(y_pred[:,:,4:], [-1, num_classes])
        
        class_true = tf.argmax(conf_true, axis=1)
        class_pred = tf.argmax(conf_pred, axis=1)
        conf = tf.reduce_max(conf_pred, axis=1)
        
        neg_mask_float = conf_true[:,0]
        neg_mask = tf.cast(neg_mask_float, tf.bool)
        pos_mask = tf.logical_not(neg_mask)
        pos_mask_float = tf.cast(pos_mask, tf.float32)
        num_total = tf.cast(tf.shape(conf_true)[0], tf.float32)
        num_pos = tf.reduce_sum(pos_mask_float)
        num_neg = num_total - num_pos
        
        conf_loss = focal_loss(conf_true, conf_pred, alpha=self.class_weights)
        conf_loss = tf.reduce_sum(conf_loss)
        
        conf_loss = conf_loss / (num_total + eps)
        
        # offset loss
        loc_true = tf.reshape(y_true[:,:,0:4], [-1, 4])
        loc_pred = tf.reshape(y_pred[:,:,0:4], [-1, 4])
        
        loc_loss = smooth_l1_loss(loc_true, loc_pred)
        pos_loc_loss = tf.reduce_sum(loc_loss * pos_mask_float) # only for positive ground truth
        
        loc_loss = pos_loc_loss / (num_pos + eps)
        
        # total loss
        total_loss = self.lambda_conf * conf_loss + self.lambda_offsets * loc_loss
        
        # metrics
        precision, recall, accuracy, fmeasure = compute_metrics(class_true, class_pred, conf, top_k=100*batch_size)
        
        def make_fcn(t):
            return lambda y_true, y_pred: t
        for name in ['conf_loss', 
                     'loc_loss', 
                     'precision', 
                     'recall',
                     'accuracy',
                     'fmeasure', 
                    ]:
            f = make_fcn(eval(name))
            f.__name__ = name
            self.metrics.append(f)
        
        return total_loss


