import numpy as np
import os
from xml.etree import ElementTree

from ssd_data import BaseGTUtility


class GTUtility(BaseGTUtility):
    """Utility for PASCAL VOC (Visual Object Classes) dataset.

    # Arguments
        data_path: Path to ground truth and image data.
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_path = os.path.join(data_path, 'JPEGImages')
        self.gt_path = gt_path = os.path.join(self.data_path, 'Annotations')
        self.classes = ['Background',
                        'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                        'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                        'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                        'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        classes_lower = [s.lower() for s in self.classes]
        
        self.image_names = []
        self.data = []
        for filename in os.listdir(gt_path):
            tree = ElementTree.parse(os.path.join(gt_path, filename))
            root = tree.getroot()
            boxes = []
            size_tree = root.find('size')
            img_width = float(size_tree.find('width').text)
            img_height = float(size_tree.find('height').text)
            image_name = root.find('filename').text
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                class_idx = classes_lower.index(class_name)
                for box in object_tree.iter('bndbox'):
                    xmin = float(box.find('xmin').text) / img_width
                    ymin = float(box.find('ymin').text) / img_height
                    xmax = float(box.find('xmax').text) / img_width
                    ymax = float(box.find('ymax').text) / img_height
                    box = [xmin, ymin, xmax, ymax, class_idx]
                    boxes.append(box)
            boxes = np.asarray(boxes)
            self.image_names.append(image_name)
            self.data.append(boxes)
        
        self.init()

    def convert_to_coco(self):
        coco_classes = [
            'Background',    
            'person',        
            'bicycle',       
            'car',           
            'motorcycle',    
            'airplane',      
            'bus',           
            'train',         
            'truck',         
            'boat',          
            'traffic light', 
            'fire hydrant',  
            'stop sign',     
            'parking meter', 
            'bench',         
            'bird',          
            'cat',           
            'dog',           
            'horse',         
            'sheep',         
            'cow',           
            'elephant',      
            'bear',          
            'zebra',         
            'giraffe',       
            'backpack',      
            'umbrella',      
            'handbag',       
            'tie',           
            'suitcase',      
            'frisbee',       
            'skis',          
            'snowboard',     
            'sports ball',   
            'kite',          
            'baseball bat',  
            'baseball glove',
            'skateboard',    
            'surfboard',     
            'tennis racket', 
            'bottle',        
            'wine glass',    
            'cup',           
            'fork',          
            'knife',         
            'spoon',         
            'bowl',          
            'banana',        
            'apple',         
            'sandwich',      
            'orange',        
            'broccoli',      
            'carrot',        
            'hot dog',       
            'pizza',         
            'donut',         
            'cake',          
            'chair',         
            'couch',         
            'potted plant',  
            'bed',           
            'dining table',  
            'toilet',        
            'tv',            
            'laptop',        
            'mouse',         
            'remote',        
            'keyboard',      
            'cell phone',    
            'microwave',     
            'oven',          
            'toaster',       
            'sink',          
            'refrigerator',  
            'book',          
            'clock',         
            'vase',          
            'scissors',      
            'teddy bear',    
            'hair drier',    
            'toothbrush',    
        ]
        
        # only for classes with different names
        voc_to_coco_map = [
            ['aeroplane',    'airplane',     ],
            ['diningtable',  'dining table', ],
            ['motorbike',    'motorcycle',   ],
            ['pottedplant',  'potted plant', ],
            ['sofa',         'couch',        ],
            ['tvmonitor',    'tv',           ],
        ]
        
        return self.convert(coco_classes, voc_to_coco_map)


if __name__ == '__main__':
    gt_util = GTUtility('data/VOC2007')
    print(gt_util.classes)
    gt = gt_util.data
    print(gt) 

