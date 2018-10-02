import numpy as np
import glob
import re
import csv

from ssd_data import BaseGTUtility

from thirdparty.get_image_size import get_image_size


class GTUtility(BaseGTUtility):
    """Utility for RoboTT-Net dataset.

    # Arguments
        data_path: Path to ground truth and image data.
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_path = data_path
        self.gt_path = data_path
        self.classes = ['Background', 'Lot 0', 'Lot 90', 'Lot 180', 'Lot -90']
        self.classes_lower = [s.lower() for s in self.classes]
        self.num_classes = len(self.classes)
        
        self.image_names = []
        self.data = []
        for filename in sorted(glob.glob(data_path+'**/*_location.csv', recursive=True)):
            image_path = re.sub(r'_location.csv$', '.jpg', filename)
            image_name = image_path[len(data_path):]
            boxes = []
            with open(filename, newline='') as f:
                reader = csv.reader(f, delimiter=';')
                header = next(reader)
                box = next(reader, None)
                if box:
                    class_idx = ['0', '90', '180', '-90'].index(box[0]) + 1
                    
                    image_size = get_image_size(image_path)
                    img_width, img_height = image_size
                    
                    xmin = float(box[2]) / img_width
                    ymin = float(box[1]) / img_height
                    xmax = float(box[4]) / img_width
                    ymax = float(box[3]) / img_height
                    
                    box = [xmin, ymin, xmax, ymax, class_idx]
                    boxes.append(box)
            
            # only images with boxes
            if len(boxes) == 0:
                continue
                boxes = np.empty((0,4+self.num_classes))
            else:
                boxes = np.asarray(boxes)
                
            self.image_names.append(image_name)
            self.data.append(boxes)
