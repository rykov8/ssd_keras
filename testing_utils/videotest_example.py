import keras
import pickle
from videotest import VideoTest

import sys
sys.path.append("..")
from ssd import SSD300 as SSD

input_shape = (300,300,3)
class_names = ["background", "spam", "eggs", "ham"] 
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)
model.load_weights('path/to/trained/weights-file.h5')
        
vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
vid_test.run('path/to/your/video.mkv')
