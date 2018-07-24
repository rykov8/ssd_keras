import numpy as np
import os

from thirdparty.get_image_size import get_image_size

from ssd_data import BaseGTUtility


class GTUtility(BaseGTUtility):
    """Utility for ICDAR2015 (International Conference on Document Analysis
    and Recognition) Focused Scene Text dataset.

    # Arguments
        data_path: Path to ground truth and image data.
        test: Boolean for using training or test set.
        polygon: Return oriented boxes defined by their four corner points.
            Required by SegLink...
    """

    def __init__(self, data_path, test=False, polygon=False):
        self.data_path = data_path
        if test:
            gt_path = os.path.join(data_path, 'Challenge2_Test_Task1_GT')
            image_path = os.path.join(data_path, 'Challenge2_Test_Task12_Images')            
        else:
            gt_path = os.path.join(data_path, 'Challenge2_Training_Task1_GT')
            image_path = os.path.join(data_path, 'Challenge2_Training_Task12_Images')
        self.gt_path = gt_path
        self.image_path = image_path
        self.classes = ['Background', 'Text']
        
        self.image_names = []
        self.data = []
        self.text = []
        for image_name in os.listdir(image_path):
            img_width, img_height = get_image_size(os.path.join(image_path, image_name))
            boxes = []
            text = []
            gt_file_name = 'gt_' + os.path.splitext(image_name)[0] + '.txt'
            with open(os.path.join(gt_path, gt_file_name), 'r') as f:
                for line in f:
                    line_split = line.strip().split(' ')
                    assert len(line_split) == 5, "length is %d" % len(line_split)
                    box = [float(v.replace(',', '')) for v in line_split[0:4]]
                    box[0] /= img_width
                    box[1] /= img_height
                    box[2] /= img_width
                    box[3] /= img_height
                    if polygon:
                        xmin = box[0]
                        ymin = box[1]
                        xmax = box[2]
                        ymax = box[3]
                        box = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                    box = box + [1]
                    boxes.append(box)
                    text.append(line_split[4][1:-1])
            boxes = np.asarray(boxes)
            self.image_names.append(image_name)
            self.data.append(boxes)
            self.text.append(text)
        
        self.init()


if __name__ == '__main__':
    gt_util = GTUtility('data/ICDAR2015_FST', test=True)
    print(gt_util.data)
