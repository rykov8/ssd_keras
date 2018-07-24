import numpy as np
import os
import glob

from thirdparty.get_image_size import get_image_size

from ssd_data import BaseGTUtility


def rot_matrix(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st],[st, ct]])


class GTUtility(BaseGTUtility):
    """Utility for MSRA-TD500 dataset.

    # Arguments
        data_path: Path to ground truth and image data.
        test: Boolean for using training or test set.
    """

    def __init__(self, data_path, test=False):
        self.data_path = data_path
        if test:
            gt_path = os.path.join(data_path, 'test')
        else:
            gt_path = os.path.join(data_path, 'train')
        self.gt_path = gt_path
        self.image_path = image_path = gt_path
        self.classes = ['Background', 'Text']
        
        self.image_names = []
        self.data = []
        self.text = []
        for image_file_name in sorted(glob.glob(image_path+'/*.JPG')):
            image_name = os.path.split(image_file_name)[1]
            img_width, img_height = get_image_size(image_file_name)
            boxes = []
            text = []
            gt_file_name = os.path.splitext(image_name)[0] + '.gt'
            with open(os.path.join(gt_path, gt_file_name), 'r') as f:
                for line in f:
                    line_split = line.strip().split(' ')
                    # line_split = [index, difficult, x, y, w, h, theta]
                    
                    # skip difficult boxes
                    if int(line_split[1]) == 1:
                        #continue
                        pass
                    
                    cx, cy, w, h, theta = [float(v) for v in line_split[2:]]
                    box = np.array([[-w,h],[w,h],[w,-h],[-w,-h]]) / 2.
                    box = np.dot(box, rot_matrix(-theta))
                    box += [cx + w/2., cy + h/2.]
                    box = list(box.flatten())
                    box = box + [1]
                    boxes.append(box)
                    text.append('')
            
            # only images with boxes
            if len(boxes) == 0:
                continue
                boxes = np.empty((0,8+self.num_classes))
            else:
                boxes = np.asarray(boxes)
            
            boxes[:,0:8:2] /= img_width
            boxes[:,1:8:2] /= img_height
            self.image_names.append(image_name)
            self.data.append(boxes)
            self.text.append(text)
            
        self.init()


if __name__ == '__main__':
    gt_util = GTUtility('data/MSRA-TD500/', test=True)
    print(gt_util.data)
