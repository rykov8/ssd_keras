"""Some utils for SSD."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import h5py
import os
import sys

from ssd_utils import PriorMap

from utils.bboxes import rot_matrix
from utils.bboxes import polygon_to_rbox, rbox_to_polygon
from utils.bboxes import polygon_to_rbox2, rbox2_to_polygon
from utils.bboxes import polygon_to_rbox3, rbox3_to_polygon

eps = 1e-10

mean = lambda x: np.sum(x)/len(x)

def plot_rbox(box, color='r', linewidth=1):
    xy_rec = rbox_to_polygon(box)
    ax = plt.gca()
    ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=linewidth))


class PriorUtil(object):
    """Utility for LinkSeg prior boxes.
    """
    def __init__(self, model, gamma=1.5):
        
        source_layers_names = [l.name.split('/')[0] for l in model.source_layers]
        self.source_layers_names = source_layers_names
        
        self.model = model
        self.image_size = model.input_shape[1:3]
        self.image_h, self.image_w = self.image_size
        self.prior_maps = []
        previous_map_size = None
        for i in range(len(source_layers_names)):
            layer = model.get_layer(source_layers_names[i])
            #map_w, map_h = map_size = layer.output_shape[1:3]
            map_h, map_w = map_size = layer.output_shape[1:3]
            if i > 0 and np.all(np.array(previous_map_size) != np.array(map_size)*2):
                print('wrong source layer size...')
            previous_map_size = map_size
            a_l = gamma * self.image_w / map_w
            m = PriorMap(source_layer_name=source_layers_names[i],
                         image_size=self.image_size,
                         map_size=map_size,
                         minmax_size=(a_l, a_l),
                         aspect_ratios=[1])
            m.a_l = m.map_size[0]
            self.prior_maps.append(m)
        self.update_priors()
    
    @property
    def num_maps(self):
        return len(self.prior_maps)
    
    def update_priors(self):
        map_offsets = [0]
        priors = []
        priors_xy = []
        priors_wh = []
        priors_variances = []
        inter_layer_neighbors_idxs = []
        cross_layer_neighbors_idxs = []
        inter_layer_neighbors_valid = []
        cross_layer_neighbors_valid = []
        
        for i in range(len(self.prior_maps)):
            m = self.prior_maps[i]
            
            # compute prior boxes
            m.compute_priors()
            
            num_priors = len(m.priors)
            
            # collect prior data
            priors.append(m.priors)
            priors_xy.append(m.priors_xy)
            priors_wh.append(m.priors_wh)
            priors_variances.append(m.priors_variances)
            
            # compute inter layer neighbors
            #w, h = m.map_size
            h, w = m.map_size
            xy_pos = np.asanyarray(np.meshgrid(np.arange(w), np.arange(h))).reshape(2,-1).T
            xy = np.tile(xy_pos, (1,8))
            xy += np.array([-1,-1, 0,-1, +1,-1, 
                            -1, 0,       +1, 0, 
                            -1,+1, 0,+1, +1,+1])
            m.inter_layer_neighbors_valid = (xy[:,0::2] >= 0) & (xy[:,0::2] < w) & (xy[:,1::2] >= 0) & (xy[:,1::2] < h)
            m.inter_layer_neighbors_idxs = xy[:,1::2] * w + xy[:,0::2]
            
            inter_layer_neighbors_idxs.append(m.inter_layer_neighbors_idxs + map_offsets[-1])
            inter_layer_neighbors_valid.append(m.inter_layer_neighbors_valid)
            
            # compute corss layer neighbors
            if i > 0:
                # previous map has always double the size of the current map
                w *= 2; h *= 2 # has to be the same as self.prior_maps[i-1].map_size
                xy = np.tile(xy_pos, (1,4))
                xy *= 2
                xy += np.array([0,0, 1,0,
                                0,1, 1,1])
                m.cross_layer_neighbors_valid = (xy[:,0::2] >= 0) & (xy[:,0::2] < w) & (xy[:,1::2] >= 0) & (xy[:,1::2] < h)
                m.cross_layer_neighbors_idxs = xy[:,1::2] * w + xy[:,0::2]
                # corss layer neighbors are always valid!
                
                cross_layer_neighbors_idxs.append(m.cross_layer_neighbors_idxs + map_offsets[-2])
                cross_layer_neighbors_valid.append(m.cross_layer_neighbors_valid)
            
            map_offsets.append(map_offsets[-1] + num_priors)
        
        self.map_offsets = map_offsets
        self.priors = np.concatenate(priors, axis=0)
        self.priors_xy = np.concatenate(priors_xy, axis=0)
        self.priors_wh = np.concatenate(priors_wh, axis=0)
        self.priors_variances = np.concatenate(priors_variances, axis=0)
        self.inter_layer_neighbors_idxs = np.concatenate(inter_layer_neighbors_idxs, axis=0)
        self.cross_layer_neighbors_idxs = np.concatenate(cross_layer_neighbors_idxs, axis=0)
        self.inter_layer_neighbors_valid = np.concatenate(inter_layer_neighbors_valid, axis=0)
        self.cross_layer_neighbors_valid = np.concatenate(cross_layer_neighbors_valid, axis=0)
        
    def encode(self, gt_data, debug=False):
        """Encode ground truth polygones to segments and links for local classification and regression.
        
        # Arguments
            gt_data: shape (boxes, 4 xy + classes)
        
        # Return
            shape (priors, 2 segment_labels + 5 segment_offsets + 2*8 inter_layer_links_labels + 2*4 cross_layer_links_labels)
        """
        
        rboxes = []
        polygons = []
        for word in gt_data:
            xy = np.reshape(word[:8], (-1, 2))
            xy = np.copy(xy) * (self.image_w, self.image_h)
            polygons.append(xy)
            rbox = polygon_to_rbox(xy)
            rboxes.append(rbox)
        rboxes = self.gt_rboxes = np.array(rboxes)
        polygnos = self.gt_polygons = np.array(polygons)
        
        # compute segments
        for i in range(len(self.prior_maps)):
            m = self.prior_maps[i]
            
            # compute priors
            #m.compute_priors()
            
            num_priors = len(m.priors)

            # assigne gt to priors
            a_l = m.minmax_size[0]
            match_indices = np.full(num_priors, -1, dtype=np.int32)
            min_lhs_eq_11 = np.full(num_priors, 1e6, dtype=np.float32)
            for j in range(len(rboxes)): # ~12.9 ms
                cx, cy, w, h, theta = rboxes[j]
                c = rboxes[j,:2]
                # constraint on ratio between box size and word height, equation (11)
                lhs_eq_11 = max(a_l/h, h/a_l)
                if lhs_eq_11 <= 1.5:
                    R = rot_matrix(theta)
                    for k in range(num_priors): # hurts
                        # is center of prior is in gt rbox
                        d = np.abs(np.dot(m.priors_xy[k]-c, R.T))
                        if d[0] < w/2. and d[1] < h/2.:
                            # is lhs of equation (11) minimal for prior
                            if lhs_eq_11 < min_lhs_eq_11[k]:
                                min_lhs_eq_11[k] = lhs_eq_11
                                match_indices[k] = j   
            m.match_indices = match_indices

            segment_mask = match_indices != -1

            # segment labels
            m.segment_labels = np.empty((num_priors, 2), dtype=np.int8)
            m.segment_labels[:, 0] = np.logical_not(segment_mask)
            m.segment_labels[:, 1] = segment_mask

            # compute offsets only for assigned boxes
            m.segment_offsets = np.zeros((num_priors, 5))
            pos_segment_idxs = np.nonzero(segment_mask)[0]
            for j in pos_segment_idxs: # box_idx # ~4 ms
                gt_idx = match_indices[j]
                rbox = rboxes[gt_idx]
                polygon = polygons[gt_idx]
                cx, cy, w, h, theta = rbox
                R = rot_matrix(theta)
                prior_x, prior_y = m.priors_xy[j]
                prior_w, prior_h = m.priors_wh[j]

                # step 2 figuer 5, rotate word anticlockwise around the center of prior
                d = rbox[:2] - m.priors_xy[j]
                poly_loc = rbox_to_polygon(list(d) + [w, h, theta])
                poly_loc_easy = polygon - m.priors_xy[j]

                poly_loc_rot = np.dot(poly_loc, R.T)

                # step 3 figure 5, crop word to left and right of prior
                poly_loc_croped = np.copy(poly_loc_rot)
                poly_loc_croped[:,0] = np.clip(poly_loc_croped[:,0], -prior_w/2., prior_w/2.)

                # step 4 figure 5, rotate croped word box clockwisely
                poly_loc_rot_back = np.dot(poly_loc_croped, R)
                rbox_loc_rot_back = polygon_to_rbox(poly_loc_rot_back)

                # encode, solve (3) to (7) to get local offsets
                offset = np.array(list(rbox_loc_rot_back[:2]/a_l) + 
                                  list(np.log(rbox_loc_rot_back[2:4]/a_l)) +
                                  [rbox_loc_rot_back[4]])
                offset[:4] /= m.priors[j,-4:] # variances
                m.segment_offsets[j] = offset
                
                # for debugging local geometry
                if debug:
                    prior_poly_loc = np.array([[-prior_w, +prior_h],
                                               [+prior_w, +prior_h],
                                               [+prior_w, -prior_h],
                                               [-prior_w, -prior_h]])/2.
                    plt.figure(figsize=[10]*2)
                    ax = plt.gca()
                    ax.add_patch(plt.Polygon(prior_poly_loc, fill=False, edgecolor='r', linewidth=1))
                    ax.add_patch(plt.Polygon(poly_loc, fill=False, edgecolor='b', linewidth=1))
                    ax.add_patch(plt.Polygon(np.dot(poly_loc, R.T), fill=False, edgecolor='k', linewidth=1))
                    #ax.add_patch(plt.Polygon(poly_loc_easy, fill=False, edgecolor='r', linewidth=1))
                    #ax.add_patch(plt.Polygon(np.dot(poly_loc_easy, R.T), fill=False, edgecolor='y', linewidth=1))
                    ax.add_patch(plt.Polygon(poly_loc_croped, fill=False, edgecolor='c', linewidth=1))
                    ax.add_patch(plt.Polygon(poly_loc_rot_back, fill=False, edgecolor='y', linewidth=1))
                    lim = 50; plt.xlim(-lim,lim); plt.ylim(-lim,lim); plt.grid()
                    plt.show()
                    debug = False

            # compute link labels
            m.inter_layer_links_labels = np.zeros((num_priors,16), dtype=np.int8)
            m.cross_layer_links_labels = np.zeros((num_priors,8), dtype=np.int8)
            if i > 0:
                previous_map = self.prior_maps[i-1]
            # we only have to check neighbors if we are positive
            for idx in pos_segment_idxs:
                neighbor_idxs = m.inter_layer_neighbors_idxs[idx]
                for n, neighbor_idx in enumerate(neighbor_idxs):
                    # valid neighbors
                    if m.inter_layer_neighbors_valid[idx,n]:
                        # neighbor matched to the same word
                        if match_indices[idx] == match_indices[neighbor_idx]:
                            # since we are positive and match to the same word, neighbor has to be positive
                            m.inter_layer_links_labels[idx, n*2+1] = 1
                # would be nice, but we refere to invalid neighbors
                #label = m.inter_layer_neighbors_valid[idx] & (match_indices[neighbor_idxs] == match_indices[idx])
                #m.inter_layer_links_labels[idx, 1::2] = label
                
                if i > 0:
                    neighbor_idxs = m.cross_layer_neighbors_idxs[idx]
                    for n, neighbor_idx in enumerate(neighbor_idxs):
                        # cross layer neighbors are always valid
                        if match_indices[idx] == previous_map.match_indices[neighbor_idx]:
                            m.cross_layer_links_labels[idx, n*2+1] = 1

            m.inter_layer_links_labels[:,::2] = np.logical_not(m.inter_layer_links_labels[:,1::2])
            m.cross_layer_links_labels[:,::2] = np.logical_not(m.cross_layer_links_labels[:,1::2])

        # collect encoded ground truth
        maps = self.prior_maps
        segment_labels = np.concatenate([m.segment_labels for m in maps])
        segment_offsets = np.concatenate([m.segment_offsets for m in maps])
        inter_layer_links_labels = np.concatenate([m.inter_layer_links_labels for m in maps])
        cross_layer_links_labels = np.concatenate([m.cross_layer_links_labels for m in maps])
        return np.concatenate([segment_labels, segment_offsets, inter_layer_links_labels, cross_layer_links_labels], axis=1)
    
    def decode(self, model_output,
                segment_threshold=0.55, link_threshold=0.35, top_k_segments=800, debug=False, debug_combining=False):
        """Decode local classification and regression results to combined bounding boxes.
        
        # Arguments
            model_output: Array with SegLink model output of shape 
                (segments, 2 segment_label + 5 segment_offset + 2*8 inter_layer_links_label 
                + 2*4 cross_layer_links_label)
            segment_threshold: Threshold for filtering segment confidence, float betwen 0 and 1.
            link_threshold: Threshold for filtering link confidence, float betwen 0 and 1.

        # Return
            Array with rboxes of shape (results, x + y + w + h + theta + confidence).
        """
        segment_labels = model_output[:,0:2]
        segment_offsets = model_output[:,2:7]
        inter_layer_links_labels = model_output[:,7:23]
        cross_layer_links_labels = model_output[:,23:31]
        
        priors_xy = self.priors_xy
        priors_wh = self.priors_wh
        priors_variances = self.priors_variances
        inter_layer_neighbors_idxs = self.inter_layer_neighbors_idxs
        cross_layer_neighbors_idxs = self.cross_layer_neighbors_idxs
        map_offsets = self.map_offsets
        first_map_offset = map_offsets[1] # 64*64
        
        # filter segments, only pos segments
        confs = segment_labels[:,1]
        segment_mask = confs > segment_threshold
        segment_mask[np.argsort(-confs)[top_k_segments:]] = False

        # filter links, pos links connected with pos segments 
        inter_layer_link_mask = (inter_layer_links_labels[:,1::2] > link_threshold) & np.repeat(segment_mask[np.newaxis, :], 8, axis=0).T
        cross_layer_link_mask = (cross_layer_links_labels[:,1::2] > link_threshold) & np.repeat(segment_mask[np.newaxis, :], 4, axis=0).T
        
        # all pos segments
        segment_idxs = np.ix_(segment_mask)[0]
        # all segments with pos links
        #inter_layer_link_idxs = np.ix_(np.logical_and.reduce(inter_layer_link_mask, axis=1))[0]
        #cross_layer_link_idxs = np.ix_(np.logical_and.reduce(cross_layer_link_mask, axis=1))[0]
        
        results = []
        
        if len(segment_idxs) > 0:
            # decode segments
            offsets = segment_offsets[segment_idxs] # delta(x,y,w,h,theta)_s
            offsets = np.copy(offsets)
            offsets[:,:4] *= priors_variances[segment_idxs] # variances

            rboxes_s = np.empty([len(offsets), 5]) # (x,y,w,h,theta)_s
            rboxes_s[:,0:2] = priors_wh[segment_idxs] * offsets[:,0:2] + priors_xy[segment_idxs]
            rboxes_s[:,2:4] = priors_wh[segment_idxs] * np.exp(np.minimum(offsets[:,2:4], 16.)) # priors_wh is filled with a_l by default
            rboxes_s[:,4] = offsets[:,4]
            rboxes_s_dict = {segment_idxs[i]: rboxes_s[i] for i in range(len(segment_idxs))}

            nodes = list(segment_idxs)
            adjacency = {n:set() for n in segment_idxs}
            for s_idx in segment_idxs:
                # collect inter layer links
                for n in np.ix_(inter_layer_link_mask[s_idx])[0]:
                    n_idx = inter_layer_neighbors_idxs[s_idx, n]
                    if n_idx in nodes:
                        # since we add only links to pos segments, they are also valid
                        adjacency[s_idx].add(n_idx)
                        adjacency[n_idx].add(s_idx)
                # collect cross layer links
                if s_idx >= first_map_offset:
                    for n in np.ix_(cross_layer_link_mask[s_idx])[0]:
                        n_idx = cross_layer_neighbors_idxs[s_idx-first_map_offset, n]
                        if n_idx in nodes:
                            adjacency[s_idx].add(n_idx)
                            adjacency[n_idx].add(s_idx)
            
            
            # find connected components
            ids = {n:None for n in segment_idxs}
            
            # recursive 
            def dfs(node, group_id):
                if ids[node] == None:
                    ids[node] = group_id
                    for a in adjacency[node]:
                        dfs(a, group_id)
            for i in range(len(nodes)):
                dfs(nodes[i], i)
            
            # none-recursive
            #stack = [*nodes]
            #while len(stack) > 0:
            #    node = stack.pop()
            #    for n in adjacency[node]:
            #        if ids[n] == None:
            #            if ids[node] == None:
            #                ids[n] = node
            #            else:
            #                ids[n] = ids[node]
            #            stack.append(n)
            
            
            groups = {i:[] for i in set(ids.values())}
            for k, v in ids.items():
                groups[v].append(k)
            
            # combine segments
            for f, k in enumerate(groups):
                # decoded segment rboxes in group
                idxs = np.array(groups[k])
                rboxes_s = np.array([rboxes_s_dict[i] for i in idxs]) # (x,y,w,h,theta)_s
                n = len(rboxes_s)
                
                # step 2, algorithm 1
                #print('rboxes_s[:,4]', rboxes_s[:,4].shape)
                theta_b = mean(rboxes_s[:,4])
                
                # step 3, algorithm 1, find minimizing b in y = a*x + b
                # minimize sum (a*x_i + b - y_i)^2 leads to b = mean(y_i - a*x_i)
                a = np.tan(-theta_b)
                a = np.copysign(np.max([np.abs(a), eps]), a) # avoid division by zero
                b = mean(rboxes_s[:,1] - a * rboxes_s[:,0])
                
                # step 4, algorithm 1, project centers on the line
                # construct line y_p = a_p*x_p + b_p that contains the point and is orthognonal to y = a*x + b
                # with a_p = -1/a and b_p = y_p - a_p * x_p we get th point of intersection
                # x_s = (b_p - b) / (a - a_p) 
                # y_s = a * x_s + b
                x_proj = (rboxes_s[:,1] + 1/a * rboxes_s[:,0] - b) / (a + 1/a)
                y_proj = a * x_proj + b
                
                # REMARK
                # set True, if you want the original SegLink decoding as described in the paper
                # the issue with the original decoding is, that step 6 makes only sense if x_p 
                # and x_q are on the left and right edge and step 8 makes only sense if x_p and
                # x_q are on the centers of the rightmost and leftmost segment
                if False:
                    # find the extreme points
                    idx_p = np.argmax(x_proj)
                    idx_q = np.argmin(x_proj)
                    x_p, y_p = x_proj[idx_q], y_proj[idx_q]
                    x_q, y_q = x_proj[idx_p], y_proj[idx_p]

                    # step 5 to 10, algorithm 1, compute the rbox values
                    w_p = rboxes_s[idx_q,2]
                    w_q = rboxes_s[idx_p,2]
                    x_b = (x_p + x_q) / 2
                    y_b = (y_p + y_q) / 2
                    w_b = ((x_p - x_q)**2 + (y_p - y_q)**2)**0.5 + (w_p + w_q) / 2
                    h_b = mean(rboxes_s[:,3])
                else:
                    idx_p = np.argmax(x_proj)
                    idx_q = np.argmin(x_proj)
                    w_p = rboxes_s[idx_p,2]
                    w_q = rboxes_s[idx_q,2]
                    x_p = rboxes_s[idx_p,0] + np.cos(theta_b) * w_p / 2
                    x_q = rboxes_s[idx_q,0] - np.cos(theta_b) * w_q / 2
                    y_p = a * x_p + b
                    y_q = a * x_q + b
                    
                    x_b = (x_p + x_q) / 2
                    y_b = (y_p + y_q) / 2
                    w_b = ((x_p - x_q)**2 + (y_p - y_q)**2)**0.5
                    h_b = mean(rboxes_s[:,3])
                
                rbox_b = [x_b, y_b, w_b, h_b, theta_b]
                
                # confidence
                confs_s = segment_labels[idxs,1]
                #conf_b = mean(confs_s)
                # weighted confidence by area of segments
                boxes_s_area = rboxes_s[:, 2]*rboxes_s[:, 3]
                conf_b = np.sum(confs_s * boxes_s_area) / np.sum(boxes_s_area)
                
                results.append(rbox_b + [conf_b])
                
                # for debugging geometric construction
                if debug_combining:
                    ax = plt.gca()
                    for rbox in rboxes_s:
                        c = 'gmbck'
                        c = c[f%len(c)]
                        plot_rbox(rbox, color=c, linewidth=1)
                        # segment centers
                        plt.plot(rbox[0], rbox[1], 'o'+c, markersize=4)
                        # projected segment centers
                        plt.plot(x_proj, y_proj, 'oy', markersize=4)
                    # lines
                    x_l = np.array([0,self.image_w])
                    y_l = a * x_l + b
                    plt.plot(x_l, y_l, 'r')
                    # endpoints
                    plt.plot(x_p, y_p, 'or', markersize=6)
                    plt.plot(x_q, y_q, 'or', markersize=6)
                    # combined box
                    plot_rbox(rbox_b, color='r', linewidth=2)

        if len(results) > 0:
            results = np.asarray(results, dtype='float32')
        else:
            results = np.empty((0,6))
        self.results = results

        # debug
        if debug:
            ax = plt.gca()
            
            # plot positive links between priors
            inter_layer_link_mask = inter_layer_links_labels[:,1::2] > link_threshold
            for idx in range(len(inter_layer_link_mask)):
                p1 = priors_xy[idx]
                n_mask = np.logical_and(inter_layer_link_mask[idx], self.inter_layer_neighbors_valid[idx])
                for n_idx in inter_layer_neighbors_idxs[idx][n_mask]:
                    p2 = priors_xy[n_idx]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y-', linewidth=2)
                
            cross_layer_link_mask = cross_layer_links_labels[:,1::2] > link_threshold
            for idx in range(len(cross_layer_neighbors_idxs)):
                p1 = priors_xy[idx+first_map_offset]
                n_mask = cross_layer_link_mask[idx+first_map_offset]
                for n_idx in cross_layer_neighbors_idxs[idx][n_mask]:
                    p2 = priors_xy[n_idx]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='orange', linewidth=2)
                    
            # plot segments
            keys = list(rboxes_s_dict.keys())
            for k in keys:
                plot_rbox(rboxes_s_dict[k], color='k', linewidth=2)

            # plot links between segments
            for k in keys:
                p1 = rboxes_s_dict[k][:2]
                for m in adjacency[k]:
                    p2 = rboxes_s_dict[m][:2]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'm-', linewidth=1)

            # plot priors
            for k in keys:
                p1 = rboxes_s_dict[k][:2]
                p2 = priors_xy[k]
                plt.plot([p1[0]], [p1[1]], 'mo', markersize=4)
                plt.plot([p2[0]], [p2[1]], 'go', markersize=4)
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1)
        
        return results
    
    def plot_neighbors(self, map_idx, location_idxs=[], inter_layer=True, cross_layer=True, color='r'):
        """Draw the linked neighborhood for given locations in a prior map.
        
        # Arguments
            map_idx: The index of the considered prior map.
            location_idxs: List of location indices in the prior map.
            inter_layer: Boolean wheter inter layer links are drawn or not.
            cross_layer: Boolean wheter cross layer links are drawn or not.
        """
        m = self.prior_maps[map_idx]
        location_idxs = [i for i in location_idxs if i < m.num_boxes]
        if inter_layer:
            for i in location_idxs:
                x, y = m.box_xy[i]
                for n_idx in m.inter_layer_neighbors_idxs[i][m.inter_layer_neighbors_valid[i]]:
                    n_x, n_y = m.box_xy[n_idx]
                    plt.plot([x, n_x], [y, n_y], '-.', color=color, linewidth=2)
        if cross_layer and map_idx > 0:
            n_m = self.prior_maps[map_idx-1]
            for i in location_idxs:
                x, y = m.box_xy[i]
                for n_idx in m.cross_layer_neighbors_idxs[i][m.cross_layer_neighbors_valid[i]]:
                    n_x, n_y = n_m.box_xy[n_idx]
                    plt.plot([x, n_x], [y, n_y], '-.', color=color, linewidth=2)
    
    def plot_gt(self):
        ax = plt.gca()
        # groud truth polygones
        for p in self.gt_polygons:
            ax.add_patch(plt.Polygon(p, fill=False, edgecolor='y', linewidth=4))
        # groud truth rboxes
        rboxes = self.gt_rboxes
        for rbox in rboxes:
            box = rbox_to_polygon(rbox)
            ax.add_patch(plt.Polygon(box, fill=False, edgecolor='b', linewidth=2))
        plt.plot(rboxes[:,0], rboxes[:,1], 'go',  markersize=4)
    
    def plot_assignment(self, map_idx):
        """Draw information about the encoded ground truth. 
        
        # Arguments
            map_idx: The index of the considered ProrMap.
        
        # Coloring    
            yellow/blue  Rotated ground truth boxes.
            magenta      Assignment of prior locations to rotatet ground truth boxes.
            cyan         Links between prior locations.
        """
        self.plot_gt()
        
        ax = plt.gca()
        rboxes = self.gt_rboxes
        m = self.prior_maps[map_idx]
        # assigned boxes
        for idx in np.nonzero(m.segment_labels[:, 1])[0]:
            p_prior = m.priors_xy[idx]
            p_word = rboxes[m.match_indices[idx]][:2]
            plt.plot([p_prior[0], p_word[0]], [p_prior[1], p_word[1]], 'm-', linewidth=1)
            #plt.plot([p_word[0]], [p_word[1]], 'ro',  markersize=8)
        # links
        labels = m.inter_layer_links_labels[:,1::2]
        idxs = np.nonzero(np.any(labels, axis=1))[0]
        for idx in idxs:
            for n_idx in m.inter_layer_neighbors_idxs[idx, np.nonzero(labels[idx])[0]]:
                x, y = m.priors_xy[idx]
                n_x, n_y = m.priors_xy[n_idx]
                plt.plot([x, n_x], [y, n_y], '-c', linewidth=1)
        if map_idx > 0:
            n_m = self.prior_maps[map_idx-1]
            labels = m.cross_layer_links_labels[:,1::2]
            idxs = np.nonzero(np.any(labels, axis=1))[0]
            for idx in idxs:
                x, y = m.priors_xy[idx]
                for n_idx in m.cross_layer_neighbors_idxs[idx, np.nonzero(labels[idx])[0]]:
                    n_x, n_y = n_m.priors_xy[n_idx]
                    plt.plot([x, n_x], [y, n_y], '-c', linewidth=1)
    
    def print_gt_stats(self):
        """Print information about the encoded ground truth""" 
        fstr = '%-5s %-5s %-5s %-5s'
        print(fstr % ('map', 'seg', 'inter', 'cross'))
        for i, m in enumerate(self.prior_maps):
            print(fstr % (i, np.sum(m.segment_labels[:,1]),
                  np.sum(m.inter_layer_links_labels[:,1::2]), 
                  np.sum(m.cross_layer_links_labels[:,1::2]) if m.cross_layer_links_labels is not None else None))

    def plot_local_evaluation(self, encoded_gt, model_output, segment_threshold=0.6, link_threshold=0.45):
        """Draw segments and links for visual evaluation.
        
        # Arguments
            ...
        
        # Color coding for segments and links
            green  True Positive
            red    False Negative
            blue   Fales Positive
        """
        
        gt_segment_mask = encoded_gt[:,1]
        gt_inter_layer_link_mask = encoded_gt[:,8:23:2]
        gt_cross_layer_links_mask = encoded_gt[:,24:31:2]
        
        segment_mask = model_output[:,1] > segment_threshold
        inter_layer_link_mask = model_output[:,8:23:2] > link_threshold
        cross_layer_link_mask = model_output[:,24:31:2] > link_threshold
        
        first_map_offset = self.map_offsets[1] # 64*64
        
        for idx in range(len(segment_mask)):
            p1 = self.priors_xy[idx]
            # segments
            g = gt_segment_mask[idx]
            p = segment_mask[idx]
            if g and p: # TP
                plt.plot(p1[0], p1[1], 'og', markersize=6)
            elif g and not p: # FN
                plt.plot(p1[0], p1[1], 'or', markersize=6)
            elif not g and p: # FP
                plt.plot(p1[0], p1[1], 'ob', markersize=6)
            # inter layer links
            for i, n_idx in enumerate(self.inter_layer_neighbors_idxs[idx]):
                if not self.inter_layer_neighbors_valid[idx, i]:
                    continue
                p2 = self.priors_xy[n_idx]
                g = gt_inter_layer_link_mask[idx,i]
                p = inter_layer_link_mask[idx,i]
                if g and p: # TP
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-g', linewidth=2)
                elif g and not p: # FN
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-r', linewidth=2)
                elif not g and p: # FP
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-b', linewidth=2)
            # cross layer links
            if idx > first_map_offset:
                for i, n_idx in enumerate(self.cross_layer_neighbors_idxs[idx-first_map_offset]):
                    p2 = self.priors_xy[n_idx]
                    g = gt_inter_layer_link_mask[idx,i]
                    p = inter_layer_link_mask[idx,i]
                    if g and p: # TP
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-g', linewidth=2)
                    elif g and not p: # FN
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-r', linewidth=2)
                    elif not g and p: # FP
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-b', linewidth=2)
    
    def plot_results(self, results=None, show_labels=False, color='r'):
        """Draw the combined bounding boxes."""
        if results is None:
            results = self.results
        ax = plt.gca()
        for r in results:
            rbox = r[:5]
            xy_rec = rbox_to_polygon(rbox)
            xy_rec = np.flip(xy_rec, axis=0) # TODO: fix this
            ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=2))
            if show_labels:
                label_name = '%.2f' % (r[5],)
                plt.text(xy_rec[0,0], xy_rec[0,1], 
                         label_name, rotation=rbox[4]/np.pi*180, 
                         bbox={'facecolor':color, 'alpha':0.5})

