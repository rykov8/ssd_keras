import numpy as np
import pyclipper
#import shapely

from sl_utils import rbox_to_polygon, polygon_to_rbox

from ssd_metric import fscore

def evaluate_results(ground_truth, detection_results, image_size=(512,512), iou_thresh=0.5):
    """Evaluate polynomial text detection results and return TP, FP, FN.
    
    # Arguments
        ground_truth: List of ground truth data with 
            shape (objects, 4 x xy + 2 x class)
        detection_results: List of corresponding detection Results with 
            shape (objects, 4 x xy + confidence + label)
        image_size: Input size of detector network.
        iou_thresh: Minimum intersection over union required to associate
            a detected polygon box to a ground truth polygon box.
    
    # Returns
        TP: True Positive detections
        FP: False Positive detections
        FN: False Negative detections
    """
    
    # we do not sort by confidence here
    # all detections are of class text
    
    gt = ground_truth
    dt = detection_results
    
    TP = []
    FP = []
    FN_sum = 0
    
    num_groundtruth_boxes = 0 # has to be TP_sum + FN_sum
    num_detections = 0
    
    for i in range(len(gt)): # samples
        #plt.figure()
        #plt.imshow(images[i])
        
        gt_polys = [np.reshape(gt[i][j,:8], (-1, 2)) * image_size for j in range(len(gt[i]))]
        dt_polys = [rbox_to_polygon(dt[i][k][:5]) for k in range(len(dt[i]))]
        
        # prepare polygones, pyclipper
        scale = 1e5
        gt_polys = [np.asarray(p*scale, dtype=np.int64) for p in gt_polys]
        dt_polys = [np.asarray(p*scale, dtype=np.int64) for p in dt_polys]
        
        # perpare polygones, shapely
        #gt_polys = [shapely.geometry.Polygon(p) for p in gt_polys]
        #dt_polys = [shapely.geometry.Polygon(p) for p in dt_polys]
        
        num_dt = len(dt_polys)
        num_gt = len(gt_polys)
        
        num_groundtruth_boxes += num_gt
        num_detections += num_dt
        
        TP_img = np.zeros(num_dt)
        FP_img = np.zeros(num_dt)
        
        assignement = np.zeros(num_gt, dtype=np.bool)
        
        for k in range(len(dt[i])): # dt
            poly1 = dt_polys[k]
            gt_iou = []
            for j in range(len(gt[i])): # gt
                poly2 = gt_polys[j]
                
                # intersection over union, pyclipper
                pc = pyclipper.Pyclipper()
                pc.AddPath(poly1, pyclipper.PT_CLIP, True)
                pc.AddPath(poly2, pyclipper.PT_SUBJECT, True)
                I = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                if len(I) > 0:
                    U = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                    Ia = pyclipper.Area(I[0])
                    Ua = pyclipper.Area(U[0])
                    IoU = Ia / Ua
                else:
                    IoU = 0.0
                
                # intersection over union, shapely, much slower
                #I = b1.intersection(b2)
                #if not I.is_empty:
                #    Ia = I.area
                #    Ua = b1.area + b2.area - Ia
                #    IoU = Ia / Ua
                #else:
                #    IoU =  0.0
                
                gt_iou.append(IoU)
                #print(IoU)
            gt_iou = np.array(gt_iou)
            max_gt_idx = np.argmax(gt_iou)
            dt_idx = k
            
            if gt_iou[max_gt_idx] > iou_thresh:
                if not assignement[max_gt_idx]: # todo: use highest iou, not first
                    TP_img[dt_idx] = 1
                    assignement[max_gt_idx] = True
                    continue
            FP_img[dt_idx] = 1
        
        FN_img_sum = np.sum(np.logical_not(assignement))
            
        TP.append(TP_img)
        FP.append(FP_img)
        FN_sum += FN_img_sum
        
        #plt.show()
        
    TP = np.concatenate(TP)
    FP = np.concatenate(FP)
    
    TP_sum = np.sum(TP)
    FP_sum = np.sum(FP)
    
    return TP_sum, FP_sum, FN_sum
    
    recall = TP_sum / (TP_sum+FN_sum)
    precision  = TP_sum / (TP_sum+FP_sum)
    print('TP %i FP %i FN %i' % (TP_sum, FP_sum, FN_sum))
    print('precision, recall, f-measure: %.2f, %.2f, %.2f' % (precision, recall, fscore(precision, recall)))

