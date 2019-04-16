import numpy as np
import os
from xml.etree import ElementTree

from ssd_data import BaseGTUtility


class GTUtility(BaseGTUtility):
    """Utility for SVT (Street View Text) dataset.

    # Arguments
        data_path: Path to ground truth and image data.
        test: Boolean for using training or test set.
        polygon: Return oriented boxes defined by their four corner points.
            Required by SegLink...
    """
    
    def __init__(self, data_path, test=False, polygon=False):
        self.data_path = data_path
        if test:
            gt_path = os.path.join(data_path, 'test.xml')
        else:
            gt_path = os.path.join(data_path, 'train.xml')
        self.gt_path = gt_path
        self.image_path = image_path = os.path.join(data_path, 'img')
        self.classes = ['Background', 'Text']
        
        self.image_names = []
        self.data = []
        self.text = []
        tree = ElementTree.parse(gt_path)
        root = tree.getroot()
        for image_tree in root.findall('image'):
            name = image_tree.find('imageName').text
            image_name = os.path.split(name)[1]
            resolution = image_tree.find('Resolution')
            img_width = float(resolution.attrib['x'])
            img_height = float(resolution.attrib['y'])
            boxes = []
            text = []
            for box_tree in image_tree.find('taggedRectangles'):
                x1 = float(box_tree.attrib['x'])
                y1 = float(box_tree.attrib['y'])
                x2 = x1 + float(box_tree.attrib['width'])
                y2 = y1 + float(box_tree.attrib['height'])
                if polygon:
                    box = [x1, y1, x1, y2, x2, y2, x2, y1, 1]
                else:
                    box = [x1, y1, x2, y2, 1]
                boxes.append(box)
                text.append(box_tree.find('tag').text)
            boxes = np.asarray(boxes)
            boxes[:,0:-1:2] /= img_width
            boxes[:,1:-1:2] /= img_height
            self.image_names.append(image_name)
            self.data.append(boxes)
            self.text.append(text)
        
        self.init()


if __name__ == '__main__':
    gt_util = GTUtility('data/SVT', test=True)
    print(gt_util.data)
