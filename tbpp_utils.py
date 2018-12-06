"""Some utils for TextBoxes++."""

import numpy as np

from ssd_utils import PriorUtil as SSDPriorUtil
from ssd_utils import iou, non_maximum_suppression, non_maximum_suppression_slow
from utils.bboxes import polygon_to_rbox3


class PriorUtil(SSDPriorUtil):
    """Utility for SSD prior boxes.
    """

    def encode(self, gt_data, overlap_threshold=0.5, debug=False):
        # calculation is done with normalized sizes
        
        # TODO: empty ground truth
        if gt_data.shape[0] == 0:
            print('gt_data', type(gt_data), gt_data.shape)
        
        num_classes = 2
        num_priors = self.priors.shape[0]
        
        gt_polygons = np.copy(gt_data[:,:8]) # normalized quadrilaterals
        gt_rboxes = np.array([polygon_to_rbox3(np.reshape(p, (-1,2))) for p in gt_data[:,:8]])
        
        # minimum horizontal bounding rectangles
        gt_xmin = np.min(gt_data[:,0:8:2], axis=1)
        gt_ymin = np.min(gt_data[:,1:8:2], axis=1)
        gt_xmax = np.max(gt_data[:,0:8:2], axis=1)
        gt_ymax = np.max(gt_data[:,1:8:2], axis=1)
        gt_boxes = self.gt_boxes = np.array([gt_xmin,gt_ymin,gt_xmax,gt_ymax]).T # normalized xmin, ymin, xmax, ymax
        
        gt_class_idx = np.asarray(gt_data[:,-1]+0.5, dtype=np.int)
        gt_one_hot = np.zeros([len(gt_class_idx),num_classes])
        gt_one_hot[range(len(gt_one_hot)),gt_class_idx] = 1 # one_hot classes including background

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
        confidence[prior_mask] = gt_one_hot[match_indices]

        gt_xy = (gt_boxes[:,2:4] + gt_boxes[:,0:2]) / 2.
        gt_wh = gt_boxes[:,2:4] - gt_boxes[:,0:2]
        gt_xy = gt_xy[match_indices]
        gt_wh = gt_wh[match_indices]
        gt_polygons = gt_polygons[match_indices]
        gt_rboxes = gt_rboxes[match_indices]
        
        priors_xy = self.priors_xy[prior_mask] / self.image_size
        priors_wh = self.priors_wh[prior_mask] / self.image_size
        variances_xy = self.priors_variances[prior_mask,0:2]
        variances_wh = self.priors_variances[prior_mask,2:4]
        
        # compute local offsets for 
        offsets = np.zeros((num_priors, 4))
        offsets[prior_mask,0:2] = (gt_xy - priors_xy) / priors_wh
        offsets[prior_mask,2:4] = np.log(gt_wh / priors_wh)
        offsets[prior_mask,0:2] /= variances_xy
        offsets[prior_mask,2:4] /= variances_wh
        
        # compute local offsets for quadrilaterals
        offsets_quads = np.zeros((num_priors, 8))
        priors_xy_minmax = np.hstack([priors_xy-priors_wh/2, priors_xy+priors_wh/2])
        #ref = np.tile(priors_xy, (1,4))
        ref = priors_xy_minmax[:,(0,1,2,1,2,3,0,3)] # corner points
        offsets_quads[prior_mask,:] = (gt_polygons - ref) / np.tile(priors_wh, (1,4)) / np.tile(variances_xy, (1,4))
        
        # compute local offsets for rotated bounding boxes
        offsets_rboxs = np.zeros((num_priors, 5))
        offsets_rboxs[prior_mask,0:2] = (gt_rboxes[:,0:2] - priors_xy) / priors_wh / variances_xy
        offsets_rboxs[prior_mask,2:4] = (gt_rboxes[:,2:4] - priors_xy) / priors_wh / variances_xy
        offsets_rboxs[prior_mask,4] = np.log(gt_rboxes[:,4] / priors_wh[:,1]) / variances_wh[:,1]
        
        return np.concatenate([offsets, offsets_quads, offsets_rboxs, confidence], axis=1)
        
        
    def decode(self, model_output, confidence_threshold=0.01, keep_top_k=200, fast_nms=True, sparse=True):
        # calculation is done with normalized sizes
        # mbox_loc, mbox_quad, mbox_rbox, mbox_conf
        # 4,8,5,2
        # boxes, quad, rboxes, confs, labels
        # 4,8,5,1,1
        
        prior_mask = model_output[:,17:] > confidence_threshold
        
        if sparse:
            # compute boxes only if the confidence is high enough and the class is not background
            mask = np.any(prior_mask[:,1:], axis=1)
            prior_mask = prior_mask[mask]
            mask = np.ix_(mask)[0]
            model_output = model_output[mask]
            priors_xy = self.priors_xy[mask] / self.image_size
            priors_wh = self.priors_wh[mask] / self.image_size
            priors_variances = self.priors_variances[mask,:]
        else:
            priors_xy = self.priors_xy / self.image_size
            priors_wh = self.priors_wh / self.image_size
            priors_variances = self.priors_variances
            
        #print('offsets', len(confidence), len(prior_mask))
        
        offsets = model_output[:,:4]
        offsets_quads = model_output[:,4:12]
        offsets_rboxs = model_output[:,12:17]
        confidence = model_output[:,17:]
        
        priors_xy_minmax = np.hstack([priors_xy-priors_wh/2, priors_xy+priors_wh/2])
        ref = priors_xy_minmax[:,(0,1,2,1,2,3,0,3)] # corner points
        variances_xy = priors_variances[:,0:2]
        variances_wh = priors_variances[:,2:4]
        
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
                #print(len(confs_to_process))
                
                # Tensorflow NMS
                #feed_dict = {
                #self.boxes: boxes_to_process,
                #    self.scores: confs_to_process
                #}
                #idx = self.sess.run(self.nms, feed_dict=feed_dict)
                
                if fast_nms:
                    idx = non_maximum_suppression(
                            boxes_to_process[:,:4], confs_to_process, 
                            self.nms_thresh, self.nms_top_k)
                else:
                    idx = non_maximum_suppression_slow(
                            boxes_to_process[:,:4], confs_to_process, 
                            self.nms_thresh, self.nms_top_k)
                
                good_boxes = boxes_to_process[idx]
                good_confs = confs_to_process[idx][:, None]
                labels = np.ones((len(idx),1)) * c
                
                good_quads = ref[mask][idx] + offsets_quads[mask][idx] * np.tile(priors_wh[mask][idx], (1,4)) * np.tile(variances_xy[mask][idx], (1,4))

                good_rboxs = offsets_rboxs[mask][idx]
                
                good_rboxs = np.empty((len(idx), 5))
                good_rboxs[:,0:2] = priors_xy[mask][idx] + offsets_rboxs[mask][idx,0:2] * priors_wh[mask][idx] * variances_xy[mask][idx]
                good_rboxs[:,2:4] = priors_xy[mask][idx] + offsets_rboxs[mask][idx,2:4] * priors_wh[mask][idx] * variances_xy[mask][idx]
                good_rboxs[:,4] = np.exp(offsets_rboxs[mask][idx,4] * variances_wh[mask][idx,1]) * priors_wh[mask][idx,1]
                
                c_pred = np.concatenate((good_boxes, good_quads, good_rboxs, good_confs, labels), axis=1)
                results.extend(c_pred)
        if len(results) > 0:
            results = np.array(results)
            order = np.argsort(-results[:, 17])
            results = results[order]
            results = results[:keep_top_k]
        else:
            results = np.empty((0,6))
        self.results = results
        return results
