"""SegLink training utils."""

import tensorflow as tf
import keras.backend as K

from ssd_training import smooth_l1_loss, softmax_loss, focal_loss
from ssd_training import compute_metrics


class SegLinkLoss(object):

    def __init__(self, lambda_segments=1.0, lambda_offsets=1.0, lambda_links=1.0, neg_pos_ratio=3.0, first_map_size=(64,64)):
        self.lambda_segments = lambda_segments
        self.lambda_offsets = lambda_offsets
        self.lambda_links = lambda_links
        self.neg_pos_ratio = neg_pos_ratio
        self.first_map_offset = first_map_size[0] * first_map_size[1] # TODO get it from model or prior_util object
        self.metrics = []
    
    def compute(self, y_true, y_pred):
        # y.shape (batches, segments, 2 x segment_label + 5 x segment_offset + 16 x inter_layer_links_label + 8 x cross_layer_links_label)
        # TODO: negatives_for_hard?
        
        batch_size = tf.shape(y_true)[0]
        eps = K.epsilon()
        
        # segment confidence loss
        seg_conf_true = tf.reshape(y_true[:,:,0:2], [-1, 2])
        seg_conf_pred = tf.reshape(y_pred[:,:,0:2], [-1, 2])
        
        seg_conf_loss = softmax_loss(seg_conf_true, seg_conf_pred)
        seg_class_pred = tf.argmax(seg_conf_pred, axis=1)
        
        
        neg_seg_mask_float = seg_conf_true[:,0]
        neg_seg_mask = tf.cast(neg_seg_mask_float, tf.bool)
        pos_seg_mask = tf.logical_not(neg_seg_mask)
        pos_seg_mask_float = tf.cast(pos_seg_mask, tf.float32)
        num_seg = tf.cast(tf.shape(seg_conf_true)[0], tf.float32)
        num_pos_seg = tf.reduce_sum(pos_seg_mask_float)
        num_neg_seg = num_seg - num_pos_seg
        
        pos_seg_conf_loss = tf.reduce_sum(seg_conf_loss * pos_seg_mask_float)
        
        
        #false_pos_seg_mask = tf.logical_and(neg_seg_mask, tf.not_equal(seg_class_pred, 0))
        #num_false_pos_seg = tf.reduce_sum(tf.cast(false_pos_seg_mask, tf.float32))
        #num_neg_seg = tf.minimum(self.neg_pos_ratio * num_pos_seg, num_false_pos_seg)
        #neg_seg_conf_loss = tf.boolean_mask(seg_conf_loss, false_pos_seg_mask)
        
        num_neg_seg = tf.minimum(self.neg_pos_ratio * num_pos_seg, num_neg_seg)
        neg_seg_conf_loss = tf.boolean_mask(seg_conf_loss, neg_seg_mask)
        
        vals, idxs = tf.nn.top_k(neg_seg_conf_loss, k=tf.cast(num_neg_seg, tf.int32))
        neg_seg_conf_loss = tf.reduce_sum(vals)
        
        
        seg_conf_loss = (pos_seg_conf_loss + neg_seg_conf_loss) / (num_pos_seg + num_neg_seg + eps)
        seg_conf_loss = self.lambda_segments * seg_conf_loss
        
        # segment offset loss
        seg_loc_true = tf.reshape(y_true[:,:,2:7], [-1, 5])
        seg_loc_pred = tf.reshape(y_pred[:,:,2:7], [-1, 5])
        
        seg_loc_loss = smooth_l1_loss(seg_loc_true, seg_loc_pred)
        pos_seg_loc_loss = tf.reduce_sum(seg_loc_loss * pos_seg_mask_float)
        
        seg_loc_loss = pos_seg_loc_loss / (num_pos_seg + eps)
        seg_loc_loss = self.lambda_offsets * seg_loc_loss
        
        # link confidence loss
        inter_link_conf_true = y_true[:,:,7:23]
        cross_link_conf_true = y_true[:,self.first_map_offset:,23:31]
        link_conf_true = tf.concat([tf.reshape(inter_link_conf_true, [-1, 2]), 
                                    tf.reshape(cross_link_conf_true, [-1, 2])], 0)
        inter_link_conf_pred = y_pred[:,:,7:23]
        cross_link_conf_pred = y_pred[:,self.first_map_offset:,23:31]
        link_conf_pred = tf.concat([tf.reshape(inter_link_conf_pred, [-1, 2]), 
                                    tf.reshape(cross_link_conf_pred, [-1, 2])], 0)
        
        
        link_conf_loss = softmax_loss(link_conf_true, link_conf_pred)
        link_class_pred = tf.argmax(link_conf_pred, axis=1)
        
        
        neg_link_mask_float = link_conf_true[:,0]
        neg_link_mask = tf.cast(neg_link_mask_float, tf.bool)
        pos_link_mask = tf.logical_not(neg_link_mask)
        pos_link_mask_float = tf.cast(pos_link_mask, tf.float32)
        num_link = tf.cast(tf.shape(link_conf_true)[0], tf.float32)
        num_pos_link = tf.reduce_sum(pos_link_mask_float)
        num_neg_link = num_link - num_pos_link
        
        pos_link_conf_loss = tf.reduce_sum(link_conf_loss * pos_link_mask_float)
        
        
        #false_pos_link_mask = tf.logical_and(neg_link_mask, tf.not_equal(link_class_pred, 0))
        #num_false_pos_link = tf.reduce_sum(tf.cast(false_pos_link_mask, tf.float32))
        #num_neg_link = tf.minimum(self.neg_pos_ratio * num_pos_link, num_false_pos_link)
        #neg_link_conf_loss = tf.boolean_mask(link_conf_loss, false_pos_link_mask)
        
        num_neg_link = tf.minimum(self.neg_pos_ratio * num_pos_link, num_neg_link)
        neg_link_conf_loss = tf.boolean_mask(link_conf_loss, neg_link_mask)
        
        vals, idxs = tf.nn.top_k(neg_link_conf_loss, k=tf.cast(num_neg_link, tf.int32))
        neg_link_conf_loss = tf.reduce_sum(vals)
        
        
        link_conf_loss = (pos_link_conf_loss + neg_link_conf_loss) / (num_pos_link + num_neg_link + eps)
        link_conf_loss = self.lambda_links * link_conf_loss
        
        # total loss
        total_loss = seg_conf_loss + seg_loc_loss + link_conf_loss
        
        
        seg_conf = tf.reduce_max(seg_conf_pred, axis=1)
        seg_class_true = tf.argmax(seg_conf_true, axis=1)
        seg_class_pred = tf.argmax(seg_conf_pred, axis=1)
        seg_precision, seg_recall, seg_accuracy, seg_fmeasure = compute_metrics(seg_class_true, seg_class_pred, seg_conf, top_k=100*batch_size)
        
        link_conf = tf.reduce_max(link_conf_pred, axis=1)
        link_class_true = tf.argmax(link_conf_true, axis=1)
        link_class_pred = tf.argmax(link_conf_pred, axis=1)
        link_precision, link_recall, link_accuracy, link_fmeasure = compute_metrics(link_class_true, link_class_pred, link_conf, top_k=100*batch_size)
        
        # metrics
        pos_seg_conf_loss = pos_seg_conf_loss / (num_pos_seg + eps)
        neg_seg_conf_loss = neg_seg_conf_loss / (num_neg_seg + eps)
        pos_link_conf_loss = pos_link_conf_loss / (num_pos_link + eps)
        neg_link_conf_loss = neg_link_conf_loss / (num_neg_link + eps)
        
        def make_fcn(t):
            return lambda y_true, y_pred: t
        for name in ['seg_conf_loss',
                     'seg_loc_loss',
                     'link_conf_loss',
                     
                     'num_pos_seg', 
                     'num_neg_seg', 
                     
                     'pos_seg_conf_loss', 
                     'neg_seg_conf_loss', 
                     'pos_link_conf_loss', 
                     'neg_link_conf_loss',
                     
                     'seg_precision', 
                     'seg_recall', 
                     'seg_accuracy', 
                     'seg_fmeasure',
                     'link_precision', 
                     'link_recall', 
                     'link_accuracy', 
                     'link_fmeasure',
                    ]:
            f = make_fcn(eval(name))
            f.__name__ = name
            self.metrics.append(f)
        
        return total_loss


class SegLinkFocalLoss(object):

    def __init__(self, lambda_segments=100.0, lambda_offsets=1.0, lambda_links=100.0, gamma_segments=2, gamma_links=2, first_map_size=(64,64)):
        self.lambda_segments = lambda_segments
        self.lambda_offsets = lambda_offsets
        self.lambda_links = lambda_links
        self.gamma_segments = gamma_segments
        self.gamma_links = gamma_links
        self.first_map_offset = first_map_size[0] * first_map_size[1] # TODO get it from model object
        self.metrics = []
    
    def compute(self, y_true, y_pred):
        # y.shape (batches, segments, 2 x segment_label + 5 x segment_offset + 16 x inter_layer_links_label + 8 x cross_layer_links_label)
        
        batch_size = tf.shape(y_true)[0]
        eps = K.epsilon()
        
        # segment confidence loss
        seg_conf_true = tf.reshape(y_true[:,:,0:2], [-1, 2])
        seg_conf_pred = tf.reshape(y_pred[:,:,0:2], [-1, 2])
        num_seg = tf.cast(tf.shape(seg_conf_true)[0], tf.float32)
        
        pos_seg_mask = seg_conf_true[:,1]
        pos_seg_mask_float = tf.cast(pos_seg_mask, tf.float32)
        num_pos_seg = tf.reduce_sum(pos_seg_mask_float)
        num_neg_seg = num_seg - num_pos_seg
        
        seg_conf_loss = focal_loss(seg_conf_true, seg_conf_pred, self.gamma_segments)
        seg_conf_loss = tf.reduce_sum(seg_conf_loss)
        seg_conf_loss = seg_conf_loss / (tf.cast(num_seg, tf.float32) + eps)
        seg_conf_loss = self.lambda_segments * seg_conf_loss
        
        # segment offset loss
        seg_loc_true = tf.reshape(y_true[:,:,2:7], [-1, 5])
        seg_loc_pred = tf.reshape(y_pred[:,:,2:7], [-1, 5])
        
        seg_loc_loss = smooth_l1_loss(seg_loc_true, seg_loc_pred)
        pos_seg_loc_loss = tf.reduce_sum(seg_loc_loss * pos_seg_mask_float)
        
        seg_loc_loss = pos_seg_loc_loss / (num_pos_seg + eps)
        seg_loc_loss = self.lambda_offsets * seg_loc_loss
        
        # link confidence loss
        inter_link_conf_true = y_true[:,:,7:23]
        cross_link_conf_true = y_true[:,self.first_map_offset:,23:31]
        link_conf_true = tf.concat([tf.reshape(inter_link_conf_true, [-1, 2]), 
                                    tf.reshape(cross_link_conf_true, [-1, 2])], 0)
        inter_link_conf_pred = y_pred[:,:,7:23]
        cross_link_conf_pred = y_pred[:,self.first_map_offset:,23:31]
        link_conf_pred = tf.concat([tf.reshape(inter_link_conf_pred, [-1, 2]), 
                                    tf.reshape(cross_link_conf_pred, [-1, 2])], 0)
        num_link = tf.shape(link_conf_true)[0]
        
        link_conf_loss = focal_loss(link_conf_true, link_conf_pred, self.gamma_links)
        link_conf_loss = tf.reduce_sum(link_conf_loss)
        link_conf_loss = link_conf_loss / (tf.cast(num_link, tf.float32) + eps)
        link_conf_loss = self.lambda_links * link_conf_loss
        
        # total loss
        total_loss = seg_conf_loss + seg_loc_loss + link_conf_loss
        
        
        seg_conf = tf.reduce_max(seg_conf_pred, axis=1)
        seg_class_true = tf.argmax(seg_conf_true, axis=1)
        seg_class_pred = tf.argmax(seg_conf_pred, axis=1)
        seg_precision, seg_recall, seg_accuracy, seg_fmeasure = compute_metrics(seg_class_true, seg_class_pred, seg_conf, top_k=100*batch_size)
        
        link_conf = tf.reduce_max(link_conf_pred, axis=1)
        link_class_true = tf.argmax(link_conf_true, axis=1)
        link_class_pred = tf.argmax(link_conf_pred, axis=1)
        link_precision, link_recall, link_accuracy, link_fmeasure = compute_metrics(link_class_true, link_class_pred, link_conf, top_k=100*batch_size)
        
        # metrics
        def make_fcn(t):
            return lambda y_true, y_pred: t
        for name in ['seg_conf_loss', 
                     'seg_loc_loss', 
                     'link_conf_loss', 
                     
                     'num_pos_seg',
                     'num_neg_seg',
                     
                     'seg_precision', 
                     'seg_recall', 
                     'seg_accuracy', 
                     'seg_fmeasure',
                     'link_precision', 
                     'link_recall', 
                     'link_accuracy', 
                     'link_fmeasure',
                    ]:
            f = make_fcn(eval(name))
            f.__name__ = name
            self.metrics.append(f)
        
        return total_loss
