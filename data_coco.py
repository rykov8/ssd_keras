import numpy as np
import os
import json

from ssd_data import BaseGTUtility


class GTUtility(BaseGTUtility):
    """Utility for Microsoft Common Objects in Context (COCO) dataset.

    # Arguments
        data_path: Path to ground truth and image data.
        validation: Boolean for using training or validation set.
    """
    
    def __init__(self, data_path, validation=False):
        self.data_path = data_path
        if validation:
            gt_path = os.path.join(data_path, 'annotations', 'instances_val2014.json')
            image_path = os.path.join(data_path, 'val2014')
        else:
            gt_path = os.path.join(data_path, 'annotations', 'instances_train2014.json')
            image_path = os.path.join(data_path, 'train2014')
        self.gt_path = gt_path
        self.image_path = image_path
        
        with open(gt_path) as f:
            data = json.load(f)
        
        classes = ['Background']
        class_map = {} # id to index
        for category in data['categories']:
            #{'id': 78, 'name': 'microwave', 'supercategory': 'appliance'}
            class_map[category['id']] = len(classes)
            classes.append(category['name'])
        self.classes = classes
        
        self.image_names = []
        self.data = []
        images = {}
        num_classes = len(self.classes)
        for image in data['images']:
            #{'coco_url': 'http://mscoco.org/images/222304',
            # 'date_captured': '2013-11-24 19:03:15',
            # 'file_name': 'COCO_val2014_000000222304.jpg',
            # 'flickr_url': 'http://farm6.staticflickr.com/5146/5639268918_7256fccf23_z.jpg',
            # 'height': 640, 'id': 222304, 'license': 1, 'width': 359}
            image['annotations'] = []
            images[image['id']] = image
            
        for annotation in data['annotations']:
            #{'area': 4776.439799999999, 'bbox': [46.72, 252.43, 75.77, 114.94], 
            # 'category_id': 44, 'id': 79459, 'image_id': 506310, 'iscrowd': 0, 
            # 'segmentation': [[48.88, 291.18, 61.79, ...]]}
            images[annotation['image_id']]['annotations'].append(annotation)
        
        bounding_boxes = {}
        for image in images.values():
            image_name = image['file_name']
            img_width = float(image['width'])
            img_height = float(image['height'])
            boxes = []
            for annotation in image['annotations']:
                bbox = annotation['bbox']
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]
                box = [x1, y1, x2, y2]
                box[0] /= img_width
                box[1] /= img_height
                box[2] /= img_width
                box[3] /= img_height
                class_idx = class_map[annotation['category_id']]
                #class_one_hot = [0] * num_classes
                #class_one_hot[class_idx] = 1
                #box = box + class_one_hot
                box = box + [class_idx]
                boxes.append(box)
            if len(boxes) == 0:
                #print(image_name)
                continue # do not add images that contain no object
                boxes = np.empty((0,4+num_classes))
            else:
                boxes = np.asarray(boxes)
            self.image_names.append(image_name)
            self.data.append(boxes)
        
        self.init()
    
    def convert_to_voc(self):
        voc_classes = [
            'Background', 
            'Aeroplane',  
            'Bicycle',    
            'Bird',       
            'Boat',       
            'Bottle',     
            'Bus',        
            'Car',        
            'Cat',        
            'Chair',      
            'Cow',        
            'Diningtable',
            'Dog',        
            'Horse',      
            'Motorbike',  
            'Person',     
            'Pottedplant',
            'Sheep',      
            'Sofa',       
            'Train',      
            'Tvmonitor',  
        ]
        
        # only for classes with different names
        coco_to_voc_map = [
            ['airplane',     'aeroplane',    ],
            ['dining table', 'diningtable',  ],
            ['motorcycle',   'motorbike',    ],
            ['potted plant', 'pottedplant',  ],
            ['couch',        'sofa',         ],
            ['tv',           'tvmonitor'     ],
        ]
        
        return self.convert(voc_classes, coco_to_voc_map)


if __name__ == '__main__':
    gt_util = GTUtility('data/COCO', validation=True)
    gt_util.print_stats()
