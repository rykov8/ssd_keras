"""Tools for model evaluation."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

from ssd_utils import iou


eps = 1e-10


def evaluate_results(ground_truth, detection_results, gt_util, iou_thresh=0.5, max_dets=None, figsize=(10,10), return_fmeasure=False):
    """Evaluates detection results, plots precision-recall curves and 
    calculates mean Average Precision.
    
    # Arguments
        ground_truth: List of ground truth data with 
            shape (objects, x1+y1+x2+y2+label)
        detection_results: List of corresponding detection Results with 
            shape (objects, x1+y1+x2+y2+confidence+label)
        gt_util: Instance of BaseGTUtility containing metadata about the 
            dataset.
        iou_thresh: Minimum intersection over union required to associate
            a detected bounding box to a ground truth bounding box.
        max_dets: Maximal number of used detections per image.
    
    # Notes
        The maximum number of detections per image can also be limited by 
        keep_top_k argument in PriorUtil.decode.
    
    """
    
    err = np.geterr()
    np.seterr(divide='ignore', invalid='ignore')
    
    gt = ground_truth
    dt = detection_results
    
    num_classes = gt_util.num_classes
    colors = gt_util.colors

    TP = []
    FP = []
    FN_sum = np.zeros(num_classes)
    num_groundtruth_boxes = np.zeros(num_classes)
    num_detections = np.zeros(num_classes)

    conf = []

    for i in range(len(gt)):
        gt_boxes = gt[i][:,:4]
        gt_labels = gt[i][:,-1].astype(np.int32)
        
        conf_img = dt[i][:,4]
        order = np.argsort(-conf_img) # sort by confidence
        order = order[:max_dets] # only max_dets detections per image
        conf.append(conf_img[order])
        dt_img = dt[i][order]
        
        dt_boxes = dt_img[:,:4]
        dt_labels = dt_img[:,-1].astype(np.int32)
        
        num_dt_img = len(dt_labels)
        TP_img = np.zeros((num_dt_img, num_classes))
        FP_img = np.zeros((num_dt_img, num_classes))
        FN_img_sum = np.zeros(num_classes)
        
        for c in range(1,num_classes):
            gt_idxs = np.argwhere(gt_labels == c)[:,0]
            dt_idxs = np.argwhere(dt_labels == c)[:,0]
            num_gt = len(gt_idxs)
            num_dt = len(dt_idxs)
            
            num_groundtruth_boxes[c] += num_gt
            num_detections[c] += num_dt
            
            assignment = np.zeros(num_gt, dtype=np.bool)
            
            if num_dt > 0:
                for dt_idx in dt_idxs:
                    if len(gt_idxs) > 0:
                        gt_iou = iou(dt_boxes[dt_idx], gt_boxes[gt_idxs])
                        max_gt_idx = np.argmax(gt_iou)
                        if gt_iou[max_gt_idx] > iou_thresh:
                            if not assignment[max_gt_idx]:
                                # true positive
                                TP_img[dt_idx, c] = 1
                                assignment[max_gt_idx] = True
                                continue
                            # false positive (multiple detections)
                        # false positive (intersection to low)
                    # false positive (no ground truth of this class)
                    FP_img[dt_idx, c] = 1
        
            FN_img_sum[c] = np.sum(np.logical_not(assignment))
        
        if False: # debug
            plt.figure(figsize=[10]*2)
            plt.imshow(images[i])
            gt_util.plot_gt(gt[i])
            for b in dt[i]:
                plot_box(b[:4], 'percent', color='b')
            plt.show()

            print('%-19s %2s %2s %2s' % ('', 'TP', 'FP', 'FN'))
            for i in range(num_classes):
                num_TP_img = np.sum(TP_img[:,i])
                num_FP_img = np.sum(FP_img[:,i])
                num_FN_img = FN_img_sum[i]
                if num_TP_img > 0 or num_FP_img > 0 or num_FN_img > 0:
                    print('%2i %-16s %2i %2i %2i' % (i, gt_util.classes[i], num_TP_img, num_FP_img, num_FN_img))

        TP.append(TP_img)
        FP.append(FP_img)
        FN_sum += FN_img_sum

    conf = np.concatenate(conf)
    order = np.argsort(-conf)
    TP = np.concatenate(TP)[order]
    FP = np.concatenate(FP)[order]

    TP_sum = np.sum(TP, axis=0)
    FP_sum = np.sum(FP, axis=0)
    
    if return_fmeasure:
        TP_sum = np.sum(TP_sum)
        FP_sum = np.sum(FP_sum)
        FN_sum = np.sum(FN_sum)
        
        recall = TP_sum / (TP_sum+FN_sum)
        precision = TP_sum / (TP_sum+FP_sum)
        fmeasure = 2 * precision * recall / (precision + recall + eps)
        
        np.seterr(**err)
        return fmeasure

    # TP + FN = num_groundtruth_boxes
    #print(np.sum(TP, axis=0) + FN_sum)
    #print(num_groundtruth_boxes)
    # TP + FP = num_detections
    #print(np.sum(TP) + np.sum(FP), len(conf))
    
    tp = np.cumsum(TP, axis=0)
    fp = np.cumsum(FP, axis=0)
    
    recall = tp / num_groundtruth_boxes
    precision  = tp / (tp+fp)
    
    # add boundary values
    mrec = np.empty((len(conf)+2, num_classes))
    mrec[0,:] = 0
    mrec[1:-1,:] = recall
    mrec[-1,:] = 1
    mpre = np.empty((len(conf)+2, num_classes))
    mpre[0,:] = 0
    mpre[1:-1,:] = np.nan_to_num(precision)
    mpre[-1,:] = 0
    
    # AP according Pascal VOC 2012
    # cummax in reverse order
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre, axis=0), axis=0), axis=0)
    AP = np.sum((mrec[1:,:]-mrec[:-1,:])*mpre[1:,:], axis=0)
    
    MAP = np.mean(AP[1:])
    
    print('%-19s %8s %8s %8s %6s' % ('Class', 'TP', 'FP', 'FN', 'AP'))
    for i in range(1, num_classes):
        print('%2i %-16s %8i %8i %8i %6.3f' % 
                (i, gt_util.classes[i], TP_sum[i], FP_sum[i], FN_sum[i], AP[i]))
    print('%-19s %8i %8i %8i %6.3f @ %g %s' % 
            ('Sum / mAP', np.sum(TP_sum), np.sum(FP_sum), np.sum(FN_sum), MAP, iou_thresh, max_dets))
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    if False: # colors
        ax.set_prop_cycle(plt.cycler('color', colors[1:]))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid()
    plt.step(mrec[:,1:], mpre[:,1:], where='pre')
    plt.legend(gt_util.classes[1:], bbox_to_anchor=(1.04,1), loc="upper left")
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
    
    np.seterr(**err)


def fscore(precision, recall, beta=1):
    """Computes the F score.
    
    The F score is the weighted harmonic mean of precision and recall.
    
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    
    With beta = 1, this is equivalent to a F-measure (F1 score). With beta < 1, 
    assigning correct classes becomes more important, and with beta > 1 the 
    metric is instead weighted towards penalizing incorrect class assignments.
    
    # Arguments
        precision: Scalar or array.
        recall: Array of same shape as precision.
        beta: Scalar.
    
    # Return
        score: Array of same shape as precision and recall.
    """
    #eps = K.epsilon()
    eps = 1e-10
    p = precision
    r = recall
    bb = beta ** 2
    score = (1 + bb) * (p * r) / (bb * p + r + eps)
    return score


def accuracy(actual, predicted):
    """ACC = (TP+TN)/num_samples"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    return (actual == predicted).sum() / float(len(actual))

def confusion_matrix(actual, predicted, num_classes, normalize=False):
    m = [[0] * num_classes for i in range(num_classes)]
    for i in range(len(actual)):
        m[actual[i]][predicted[i]] += 1
    m = np.array(m)
    if normalize:
        m = m / np.max(m)
    return m

def plot_confusion_matrix(confusion_matrix, classes, figsize=(10,10)):
    plt.figure(figsize=figsize)
    cmap = plt.cm.Blues
    n = len(classes)
    ticks = np.arange(n)
    plt.matshow(confusion_matrix, cmap=cmap, fignum=1)
    plt.xticks(ticks, classes, rotation=90)
    plt.yticks(ticks, classes, rotation=0)
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(ticks, ticks):
        plt.text(j, i, np.round(confusion_matrix[i, j], 3),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=0.1, pad=0.05)
    plt.colorbar(cax=cax)
    plt.show()
