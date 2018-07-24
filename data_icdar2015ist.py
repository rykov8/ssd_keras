import numpy as np
import os
import glob

from thirdparty.get_image_size import get_image_size

from ssd_data import BaseGTUtility


class GTUtility(BaseGTUtility):
    """Utility for ICDAR2015 (International Conference on Document Analysis
    and Recognition) Incidental Scene Text dataset.

    # Arguments
        data_path: Path to ground truth and image data.
        test: Boolean for using training or test set.
    """

    def __init__(self, data_path, test=False):
        self.data_path = data_path
        if test:
            gt_path = os.path.join(data_path, 'ch4_test_localization_transcription_gt')
            image_path = os.path.join(data_path, 'ch4_test_images')
        else:
            gt_path = os.path.join(data_path, 'ch4_training_localization_transcription_gt')
            image_path = os.path.join(data_path, 'ch4_training_images')
        self.gt_path = gt_path
        self.image_path = image_path
        self.classes = ['Background', 'Text']
        
        self.image_names = []
        self.data = []
        self.text = []
        for image_file_name in sorted(glob.glob(image_path+'/*.jpg')):
            image_name = os.path.split(image_file_name)[1]
            img_width, img_height = get_image_size(image_file_name)
            boxes = []
            text = []
            gt_file_name = 'gt_' + os.path.splitext(image_name)[0] + '.txt'
            with open(os.path.join(gt_path, gt_file_name), 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line_split = line.strip().split(',')
                    box = [float(v) for v in line_split[0:8]]
                    box = box + [1]
                    boxes.append(box)
                    text.append(line_split[8])
            boxes = np.asarray(boxes)
            boxes[:,0:8:2] /= img_width
            boxes[:,1:8:2] /= img_height
            self.image_names.append(image_name)
            self.data.append(boxes)
            self.text.append(text)
        
        self.init()


if __name__ == '__main__':
    gt_util = GTUtility('data/ICDAR2015_IST', test=True)
    print(gt_util.data)
