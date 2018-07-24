import numpy as np
import json
import os

from ssd_data import BaseGTUtility


class GTUtility(BaseGTUtility):
    """Utility for COCO-Text dataset.

    # Arguments
        data_path: Path to ground truth and image data.
        validation: Boolean for using training or validation set.
        polygon: Return oriented boxes defined by their four corner points.
            Required by SegLink...
        only_with_label: Add only boxes if text labels are available.
    """
    
    def __init__(self, data_path, validation=False, polygon=False, only_with_label=True):
        test=False
        
        self.data_path = data_path
        gt_path = data_path
        image_path = os.path.join(data_path, 'train2014')
        self.gt_path = gt_path
        self.image_path = image_path
        self.classes = ['Background', 'Text']
        
        self.image_names = []
        self.data = []
        self.text = []
        
        with open(os.path.join(gt_path, 'COCO_Text.json')) as f:
            gt_data = json.load(f)

        for img_id in gt_data['imgToAnns'].keys(): # images
            
            ann_ids = gt_data['imgToAnns'][img_id]

            if len(ann_ids) > 0:
                img_data = gt_data['imgs'][img_id]
                set_type = img_data['set']
            
                if test:
                    if set_type != 'test':
                        continue
                elif validation:
                    if set_type != 'val':
                        continue
                else:
                    if set_type != 'train':
                        continue

                image_name = img_data['file_name']
                img_width = img_data['width']
                img_height = img_data['height']

                boxes = []
                text = []
            
                for ann_id in ann_ids: # boxes
                    ann_data = gt_data['anns'][str(ann_id)]
            
                    if polygon:
                        box = np.array(ann_data['polygon'], dtype=np.float32)
                    else:
                        x, y, w, h = np.array(ann_data['bbox'], dtype=np.float32)
                        box = np.array([x, y, x+w, y+h])
                    
                    if 'utf8_string' in ann_data.keys():
                        txt = ann_data['utf8_string']
                    else:
                        if only_with_label:
                            continue
                        else:
                            txt = ''
                    
                    boxes.append(box)
                    text.append(txt)
                
                if len(boxes) == 0:
                    continue
                
                boxes = np.asarray(boxes)
                
                boxes[:,0::2] /= img_width
                boxes[:,1::2] /= img_height
                    
                # append classes
                boxes = np.concatenate([boxes, np.ones([boxes.shape[0],1])], axis=1)
                
                self.image_names.append(image_name)
                self.data.append(boxes)
                self.text.append(text)
                
        self.init()


if __name__ == '__main__':
    gt_util = GTUtility('data/COCO-Text', validation=False, polygon=True, only_with_label=True)
    print(gt_util.data)
