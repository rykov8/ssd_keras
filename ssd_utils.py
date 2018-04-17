"""Some utils for SSD."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import h5py
import cv2
import os

from ssd_viz import to_rec


def iou(box, priors):
    """Compute intersection over union for the box with all priors.

    # Arguments
        box: Box, numpy tensor of shape (4,).
            (x1 + y1 + x2 + y2)
        priors: 

    # Return
        iou: Intersection over union,
            numpy tensor of shape (num_priors).
    """
    # compute intersection
    inter_upleft = np.maximum(priors[:, :2], box[:2])
    inter_botright = np.minimum(priors[:, 2:4], box[2:])
    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # compute union
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (priors[:, 2] - priors[:, 0]) * (priors[:, 3] - priors[:, 1])
    union = area_pred + area_gt - inter
    # compute iou
    iou = inter / union
    return iou

def non_maximum_suppression_slow(boxes, confs, iou_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.
    
    Intuitive but slow as hell!!!
    
    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
        iou_threshold: Intersection over union threshold used for comparing 
            overlapping boxes.
        top_k: Maximum number of returned indices.
    
    # Return
        List of remaining indices.
    """
    idxs = np.argsort(-confs)
    selected = []
    for idx in idxs:
        if np.any(iou(boxes[idx], boxes[selected]) >= iou_threshold):
            continue
        selected.append(idx)
        if len(selected) >= top_k:
            break
    return selected

def non_maximum_suppression(boxes, confs, overlap_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.
    
    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
        overlap_threshold: 
        top_k: Maximum number of returned indices.
    
    # Return
        List of remaining indices.
    
    # References
        - Girshick, R. B. and Felzenszwalb, P. F. and McAllester, D.
          [Discriminatively Trained Deformable Part Models, Release 5](http://people.cs.uchicago.edu/~rbg/latent-release5/)
    """
    eps = 1e-15
    
    boxes = boxes.astype(np.float64)

    pick = []
    x1, y1, x2, y2 = boxes.T
    
    idxs = np.argsort(confs)
    area = (x2 - x1) * (y2 - y1)
    
    while len(idxs) > 0:
        i = idxs[-1]
        
        pick.append(i)
        if len(pick) >= top_k:
            break
        
        idxs = idxs[:-1]
        
        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / (area[idxs] + eps)
        # TODO: this is actually not the IoU, find out why it works?
        
        idxs = idxs[overlap <= overlap_threshold]
        
    return pick



class PriorMap(object):
    """Handles prior boxes for a given feature map.
    
    # Arguments / Attributes
        source_layer_name
        image_size
        map_size
        variances
        aspect_ratios: List of aspect ratios for the prior boxes at each 
            location.
        shift: List of tuples for the displacement of the prior boxes 
            relative to ther location. Each tuple contains an value between 
            -1.0 and 1.0 for x and y direction.
        flip: Boolean, whether the reciprocal value of the aspect ratios 
            should also be used.
        clip: Boolean, whether the boxes should be cropped to do not exceed 
            the borders of the input image.
        step
        minmax_size
        special_ssd_box: Boolean, wether or not the extra box for aspect 
            ratio 1 is used.
    
    # Notes
        The compute_priors methode has to be called to get usable prior boxes.
    """
    def __init__(self, source_layer_name, image_size, map_size, 
                 minmax_size=None, variances=[0.1, 0.1, 0.2, 0.2], 
                 aspect_ratios=[1], shift=None,
                 flip=True, clip=False, step=None, special_ssd_box=False):
        self.source_layer_name = source_layer_name
        self.image_size = image_size # is input size
        self.map_size = map_size
        self.variances = variances
        self.aspect_ratios = aspect_ratios
        self.shift = shift
        self.flip = flip
        self.clip = clip
        self.step = step
        self.minmax_size = minmax_size # s_min, s_max
        self.special_ssd_box = special_ssd_box
        
        #self.compute_priors()
        
    def __str__(self):
        s = ''
        for a in ['source_layer_name',
                  'map_size',
                  'aspect_ratios',
                  'shift',
                  'clip',
                  'flip',
                  'minmax_size',
                  'special_ssd_box',
                  'num_locations',
                  'num_boxes',
                  'num_boxes_per_location',
                  ]:
            s += '%-24s %s\n' % (a, getattr(self, a))
        return s
    
    @property
    def num_boxes_per_location(self):
        return len(self.box_wh)
    
    @property
    def num_locations(self):
        return len(self.box_xy)
    
    @property
    def num_boxes(self):
        return len(self.box_xy) * len(self.box_wh) # len(self.priors)
    
    def compute_priors(self):
        image_h, image_w = self.image_size
        map_h, map_w = self.map_size
        min_size, max_size = self.minmax_size
        
        # define centers of prior boxes
        if self.step == None:
            step_x = image_w / map_w
            step_y = image_h / map_h
            linx = np.linspace(step_x / 2., image_w - step_x / 2., map_w)
            liny = np.linspace(step_y / 2., image_h - step_y / 2., map_h)
        else:
            # for compatibility with caffe models
            step_x = step_y = self.step
            linx = np.array([(0.5 + i) for i in range(map_w)]) * step_x
            liny = np.array([(0.5 + i) for i in range(map_h)]) * step_y
        box_xy = np.array(np.meshgrid(linx, liny)).reshape(2,-1).T
        
        if self.shift is None:
            shift = [(0.0,0.0)] * len(self.aspect_ratios)
        else:
            shift = self.shift
        
        box_wh = []
        box_shift = []
        for i in range(len(self.aspect_ratios)):
            ar = self.aspect_ratios[i]
            box_wh.append([min_size * np.sqrt(ar), min_size / np.sqrt(ar)])
            box_shift.append(shift[i])
            if ar == 1 and self.special_ssd_box: # special SSD box
                box_wh.append([np.sqrt(min_size * max_size), np.sqrt(min_size * max_size)])
                box_shift.append((0.0,0.0))
            if not ar == 1 and self.flip:
                box_wh.append([min_size * np.sqrt(1.0/ar), min_size / np.sqrt(1.0/ar)])
                box_shift.append(shift[i])
        box_wh = np.asarray(box_wh)
        
        box_shift = np.asarray(box_shift)
        box_shift = np.clip(box_shift, -1.0, 1.0)
        box_shift = box_shift * 0.5 * np.array([step_x, step_y]) # percent to pixels
        
        # values for individual prior boxes
        priors_shift = np.tile(box_shift, (len(box_xy),1))
        priors_xy = np.repeat(box_xy, len(box_wh), axis=0) + priors_shift
        priors_wh = np.tile(box_wh, (len(box_xy),1))
                
        priors_min_xy = priors_xy - priors_wh / 2.
        priors_max_xy = priors_xy + priors_wh / 2.
        
        if self.clip:
            priors_min_xy[:,0] = np.clip(priors_min_xy[:,0], 0, image_w)
            priors_min_xy[:,1] = np.clip(priors_min_xy[:,1], 0, image_h)
            priors_max_xy[:,0] = np.clip(priors_max_xy[:,0], 0, image_w)
            priors_max_xy[:,1] = np.clip(priors_max_xy[:,1], 0, image_h)
        
        priors_variances = np.tile(self.variances, (len(priors_xy),1))
        
        priors = np.concatenate([priors_min_xy, priors_max_xy, priors_variances], axis=1)
        
        # normalized prior boxes
        priors_min_xy_norm = priors_min_xy / (image_w, image_h)
        priors_max_xy_norm = priors_max_xy / (image_w, image_h)
        
        priors_norm = np.concatenate([priors_min_xy_norm, priors_max_xy_norm, priors_variances], axis=1)
        
        self.box_xy = box_xy
        self.box_wh = box_wh
        self.box_shfit = box_shift
        
        self.priors_xy = priors_xy
        self.priors_wh = priors_wh
        self.priors_variances = priors_variances
        self.priors = priors
        self.priors_norm = priors_norm
    
    def plot_locations(self):
        xy = self.box_xy
        plt.plot(xy[:,0], xy[:,1], 'r.', markersize=4)
    
    def plot_boxes(self, location_idxs=[]):
        colors = 'rgbcmy'*3
        ax = plt.gca()
        n = self.num_boxes_per_location
        for i in location_idxs:
            for j in range(n):
                idx = i*n+j
                if idx >= self.num_boxes:
                    break
                x1, y1, x2, y2 = self.priors[idx, :4]
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                        fill=False, edgecolor=colors[j], linewidth=2))
        ax.autoscale_view()


class PriorUtil(object):
    """Utility for SSD prior boxes.
    """
    def __init__(self, model, source_layers_names=None, aspect_ratios=None, shifts=None,
            minmax_sizes=None, steps=None, special_ssd_boxes=None, flips=None, clips=None):
        
        self.model = model
        self.image_size = model.input_shape[1:3]
        
        # take parameters from model definition if they exist there
        if source_layers_names is None:
            if hasattr(model, 'source_layers_names'):
                source_layers_names = model.source_layers_names
        num_maps = len(source_layers_names)
        if aspect_ratios is None:
            if hasattr(model, 'aspect_ratios'):
                aspect_ratios = model.aspect_ratios
            else:
                aspect_ratios = [[1]] * num_maps
        if shifts is None:
            if hasattr(model, 'shifts'):
                shifts = model.shifts
            else:
                shifts = [None] * num_maps
        if minmax_sizes is None:
            if hasattr(model, 'minmax_sizes'):
                minmax_sizes = model.minmax_sizes
            else:
                # as in equation (4)
                min_dim = np.min(self.image_size)
                min_ratio = 10 # 15
                max_ratio = 100 # 90
                s = np.linspace(min_ratio, max_ratio, num_maps+1) * min_dim / 100.
                minmax_sizes = [(round(s[i]), round(s[i+1])) for i in range(len(s)-1)]
        if steps is None:
            if hasattr(model, 'steps'):
                steps = model.steps
            else:
                steps = [None] * num_maps
        if special_ssd_boxes is None:
            if hasattr(model, 'special_ssd_boxes'):
                special_ssd_boxes = model.special_ssd_boxes
            else:
                special_ssd_boxes = True
        if type(special_ssd_boxes) == bool:
            special_ssd_boxes = [special_ssd_boxes] * num_maps
        if flips is None:
            if hasattr(model, 'flips'):
                flips = model.flips
            else:
                flips = True
        if type(flips) == bool:
            flips = [flips] * num_maps
        if clips is None:
            if hasattr(model, 'clips'):
                clips = model.clips
            else:
                clips = False
        if type(clips) == bool:
            clips = [clips] * num_maps
        
        self.prior_maps = []
        for i in range(num_maps):
            layer = model.get_layer(source_layers_names[i])
            map_h, map_w = map_size = layer.output_shape[1:3]
            m = PriorMap(source_layer_name=source_layers_names[i],
                         image_size=self.image_size,
                         map_size=map_size,
                         minmax_size=minmax_sizes[i],
                         variances=[0.1, 0.1, 0.2, 0.2],
                         aspect_ratios=aspect_ratios[i],
                         shift=shifts[i],
                         step=steps[i],
                         special_ssd_box=special_ssd_boxes[i],
                         flip=flips[i],
                         clip=clips[i])
            self.prior_maps.append(m)
        self.update_priors()
        
        self.nms_top_k = 400
        self.nms_thresh = 0.45
        
        # Tensorflow NMS
        #self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        #self.scores = tf.placeholder(dtype='float32', shape=(None,))
        #self.nms = tf.image.non_max_suppression(
        #    self.boxes, self.scores, self.nms_top_k, 
        #    iou_threshold=self.nms_thresh)
        #self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    
    @property
    def num_maps(self):
        return len(self.prior_maps)
    
    def update_priors(self):
        priors_xy = []
        priors_wh = []
        priors = []
        priors_norm = []
        map_offsets = [0]
        for i in range(len(self.prior_maps)):
            m = self.prior_maps[i]
            
            # compute prior boxes
            m.compute_priors()
            
            # collect prior data
            priors_xy.append(m.priors_xy)
            priors_wh.append(m.priors_wh)
            priors.append(m.priors)
            priors_norm.append(m.priors_norm)
            map_offsets.append(map_offsets[-1]+len(m.priors))
        
        self.priors_xy = np.concatenate(priors_xy, axis=0)
        self.priors_wh = np.concatenate(priors_wh, axis=0)
        self.priors = np.concatenate(priors, axis=0)
        self.priors_norm = np.concatenate(priors_norm, axis=0)
        self.map_offsets = map_offsets
        
    def encode(self, gt_data, overlap_threshold=0.5, debug=False):
        # calculation is done with normalized sizes

        gt_boxes = self.gt_boxes = np.copy(gt_data[:,:4]) # normalized xmin, ymin, xmax, ymax
        gt_labels = self.gt_labels = np.copy(gt_data[:,4:]) # one_hot classes including background

        num_priors = self.priors.shape[0]
        num_classes = gt_labels.shape[1]

        # TODO: empty ground truth
        if gt_data.shape[0] == 0:
            print('gt_data', type(gt_data), gt_data.shape)

        gt_iou = np.array([iou(b, self.priors_norm) for b in gt_boxes]).T
        
        # assigne gt to priors
        max_idxs = np.argmax(gt_iou, axis=1)
        max_val = gt_iou[np.arange(num_priors), max_idxs]
        prior_mask = max_val > overlap_threshold
        match_indices = max_idxs[prior_mask]

        self.match_indices = dict(zip(list(np.ix_(prior_mask)[0]), list(match_indices)))

        # prior labels
        confidence = np.zeros((num_priors, num_classes))
        confidence[:,0] = 1
        confidence[prior_mask] = gt_labels[match_indices]

        # compute local offsets from ground truth boxes
        gt_xy = (gt_boxes[:,2:4] + gt_boxes[:,0:2]) / 2.
        gt_wh = gt_boxes[:,2:4] - gt_boxes[:,0:2]
        gt_xy = gt_xy[match_indices]
        gt_wh = gt_wh[match_indices]
        priors_xy = self.priors_xy[prior_mask] / self.image_size
        priors_wh = self.priors_wh[prior_mask] / self.image_size
        offsets = np.zeros((num_priors, 4))
        offsets[prior_mask, 0:2] = (gt_xy - priors_xy) / priors_wh
        offsets[prior_mask, 2:4] = np.log(gt_wh / priors_wh)
        offsets[prior_mask, :] /= self.priors[prior_mask,-4:] # variances

        return np.concatenate([offsets, confidence], axis=1)
        
    def decode(self, model_output, confidence_threshold=0.01, keep_top_k=200, fast_nms=True, sparse=True):
        # calculation is done with normalized sizes
        
        prior_mask = model_output[:,4:] > confidence_threshold
        
        if sparse:
            # compute boxes only if the confidence is high enough and the class is not background
            mask = np.any(prior_mask[:,1:], axis=1)
            prior_mask = prior_mask[mask]
            mask = np.ix_(mask)[0]
            model_output = model_output[mask]
            priors_xy = self.priors_xy[mask] / self.image_size
            priors_wh = self.priors_wh[mask] / self.image_size
            priors_variances = self.priors[mask,-4:]
        else:
            priors_xy = self.priors_xy / self.image_size
            priors_wh = self.priors_wh / self.image_size
            priors_variances = self.priors[:,-4:]
        #print('offsets', len(confidence), len(prior_mask))
        
        offsets = model_output[:,:4]
        confidence = model_output[:,4:]
        
        num_priors = offsets.shape[0]
        num_classes = confidence.shape[1]

        # compute bounding boxes from local offsets
        boxes = np.empty((num_priors, 4))
        offsets = offsets * priors_variances
        boxes_xy = priors_xy + offsets[:,0:2] * priors_wh
        boxes_wh = priors_wh * np.exp(offsets[:,2:4])
        boxes[:,0:2] = boxes_xy - boxes_wh / 2. # xmin, ymin
        boxes[:,2:4] = boxes_xy + boxes_wh / 2. # xmax, ymax
        boxes = np.clip(boxes, 0.0, 1.0)
        
        # do non maximum suppression
        results = []
        for c in range(1, num_classes):
            mask = prior_mask[:,c]
            boxes_to_process = boxes[mask]
            if len(boxes_to_process) > 0:
                confs_to_process = confidence[mask, c]
                
                # Tensorflow NMS
                #feed_dict = {
                #self.boxes: boxes_to_process,
                #    self.scores: confs_to_process
                #}
                #idx = self.sess.run(self.nms, feed_dict=feed_dict)
                
                if fast_nms:
                    idx = non_maximum_suppression(
                            boxes_to_process, confs_to_process, 
                            self.nms_thresh, self.nms_top_k)
                else:
                    idx = non_maximum_suppression_slow(
                            boxes_to_process, confs_to_process, 
                            self.nms_thresh, self.nms_top_k)
                
                good_boxes = boxes_to_process[idx]
                good_confs = confs_to_process[idx][:, None]
                labels = np.ones((len(idx),1)) * c
                c_pred = np.concatenate((good_boxes, good_confs, labels), axis=1)
                results.extend(c_pred)
        if len(results) > 0:
            results = np.array(results)
            order = np.argsort(-results[:, 4])
            results = results[order]
            results = results[:keep_top_k]
        else:
            results = np.empty((0,6))
        self.results = results
        return results
    
    def show_image(self, img):
        """Resizes an image to the network input size and shows it in the current figure.
        """
        img_h, img_w = self.image_size
        img = cv2.resize(img, (img_w, img_h), cv2.INTER_LINEAR)
        img = img[:, :, (2,1,0)] # BGR to RGB
        img = img / 256.
        plt.imshow(img)
    
    def plot_assignement(self, map_idx):
        ax = plt.gca()
        im = plt.gci()
        image_height, image_width = image_size = im.get_size()
        
        # ground truth
        boxes = self.gt_boxes
        boxes_x = (boxes[:,0] + boxes[:,2]) / 2. * image_height
        boxes_y = (boxes[:,1] + boxes[:,3]) / 2. * image_width
        for box in boxes:
            xy_rec = to_rec(box[:4], image_size)
            ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor='b', linewidth=2))
        plt.plot(boxes_x, boxes_y, 'bo',  markersize=8)
        
        # prior boxes
        for idx, box_idx in self.match_indices.items():
            if idx >= self.map_offsets[map_idx] and idx < self.map_offsets[map_idx+1]:
                x, y = self.priors_xy[idx]
                w, h = self.priors_wh[idx]
                plt.plot(x, y, 'ro',  markersize=4)
                plt.plot([x, boxes_x[box_idx]], [y, boxes_y[box_idx]], '-r', linewidth=1)
                ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w+1, h+1, 
                        fill=False, edgecolor='y', linewidth=2))

    def plot_results(self, results=None, classes=None, show_labels=True, gt_data=None, confidence_threshold=None):
        if results is None:
            results = self.results
        if confidence_threshold is not None:
            mask = results[:, 4] > confidence_threshold
            results = results[mask]
        if classes is not None:
            colors = plt.cm.hsv(np.linspace(0, 1, len(classes)+1)).tolist()
        ax = plt.gca()
        im = plt.gci()
        image_size = im.get_size()
        
        # draw ground truth
        if gt_data is not None:
            for box in gt_data:
                label = np.nonzero(box[4:])[0][0]+1
                color = 'g' if classes == None else colors[label]
                xy_rec = to_rec(box[:4], image_size)
                ax.add_patch(plt.Polygon(xy_rec, fill=True, color=color, linewidth=1, alpha=0.3))
        
        # draw prediction
        for r in results:
            label = int(r[5])
            confidence = r[4]
            color = 'r' if classes == None else colors[label]
            xy_rec = to_rec(r[:4], image_size)
            ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=2))
            if show_labels:
                label_name = label if classes == None else classes[label]
                xmin, ymin = xy_rec[0]
                display_txt = '%0.2f, %s' % (confidence, label_name)        
                ax.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        
    def print_gt_stats(self):
        # TODO
        pass
    
    # TODO: multiScaleOutput


def load_weights(model, filepath, layer_names=None):
    """Loads layer weights from a HDF5 save file.
     
    # Arguments
        model: Keras model
        filepath: Path to HDF5 file
        layer_names: List of strings, names of the layers for which the 
            weights should be loaded. List of tuples, if the names in 
            the file differ from those in model.
    """
    filepath = os.path.expanduser(filepath)
    f = h5py.File(filepath, 'r')
    if layer_names == None:
        layer_names = [s.decode() for s in f.attrs['layer_names']]
    for name in layer_names:
        if type(name) in [tuple, list]:
            layer_name = name[1]
            name = name[0]
        else:
            layer_name = name
        try:
            layer = model.get_layer(layer_name)
            #assert layer is not None
        except:
            print('layer missing %s' % (layer_name))
            continue
        g = f[name]
        weights = [np.array(g[wn]) for wn in g.attrs['weight_names']]
        try:
            #print('load %s' % (layer_name))
            layer.set_weights(weights)
        except Exception as e:
            print('something went wrong %s' % (layer_name))
            print('    model %s' % ([w.shape.as_list() for w in layer.weights]))
            print('    file  %s' % ([w.shape for w in weights]))
            print(e)
    f.close()


def calc_memory_usage(model, batch_size=1):
    """Compute the memory usage of a keras modell.
    
    # Arguments
        model: Keras model.
        batch_size: Batch size used for training.
    
    source: https://stackoverflow.com/a/46216013/445710
    """

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    
    # each shape unit occupies 4 bytes in memory
    total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    
    for s in ['Byte', 'KB', 'MB', 'GB', 'TB']:
        if total_memory > 1024:
            total_memory /= 1024
        else:
            break
    print('model memory usage %.2f %s' % (total_memory, s))

