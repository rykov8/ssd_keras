"""TextBoxes++ training utils."""

import tensorflow as tf
import keras.backend as K

from ssd_training import smooth_l1_loss, focal_loss, compute_metrics


class TBPPFocalLoss(object):

    def __init__(self, lambda_conf=1000.0, lambda_offsets=1.0):
        self.lambda_conf = lambda_conf
        self.lambda_offsets = lambda_offsets
        self.metrics = []
    
    def compute(self, y_true, y_pred):
        # y.shape (batches, priors, 4 x bbox_offset + 8 x quadrilaterals + 5 x rbbox_offsets + n x class_label)
        
        batch_size = tf.shape(y_true)[0]
        num_priors = tf.shape(y_true)[1]
        num_classes = tf.shape(y_true)[2] - 17
        eps = K.epsilon()
        
        # confidence loss
        conf_true = tf.reshape(y_true[:,:,17:], [-1, num_classes])
        conf_pred = tf.reshape(y_pred[:,:,17:], [-1, num_classes])
        
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
        
        conf_loss = focal_loss(conf_true, conf_pred, alpha=[0.002, 0.998])
        conf_loss = tf.reduce_sum(conf_loss)
        
        conf_loss = conf_loss / (num_total + eps)
        
        # offset loss, bbox, quadrilaterals, rbbox
        loc_true = tf.reshape(y_true[:,:,0:17], [-1, 17])
        loc_pred = tf.reshape(y_pred[:,:,0:17], [-1, 17])
        
        loc_loss = smooth_l1_loss(loc_true, loc_pred)
        #loc_loss = smooth_l1_loss(loc_true[:,:4], loc_pred[:,:4])
        pos_loc_loss = tf.reduce_sum(loc_loss * pos_mask_float) # only for positives
        
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
                     'num_pos',
                     'num_neg'
                    ]:
            f = make_fcn(eval(name))
            f.__name__ = name
            self.metrics.append(f)
        
        return total_loss
