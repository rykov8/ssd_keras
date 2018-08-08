""" A class for testing a SSD model on a video file or webcam """

import numpy as np
import matplotlib.pyplot as plt
import cv2
from timeit import default_timer as timer

from ssd_data import preprocess


class VideoTest(object):
    """ Class for testing a trained SSD model on a video file and show the
        result in a window. Class is designed so that one VideoTest object 
        can be created for a model, and the same object can then be used on 
        multiple videos and webcams.
        
        Arguments:
            class_names: A list of strings, each containing the name of a class.
                         The first name should be that of the background class
                         which is not used.
                         
            model:       An SSD model. It should already be trained for 
                         images similar to the video to test on.
                         
            input_shape: The shape that the model expects for its input, 
                         as a tuple, for example (300, 300, 3)    
                         
            bbox_util:   An instance of the BBoxUtility class in ssd_utils.py
                         The BBoxUtility needs to be instantiated with 
                         the same number of classes as the length of        
                         class_names.
    
    """
    
    def __init__(self, model, prior_util, class_names, input_shape):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = model
        self.input_shape = input_shape
        self.prior_util = prior_util
        
        # use same colors as in plots
        colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes))
        colors = (colors[:,(2,1,0)]*255).astype(np.int32).tolist()
        self.class_colors = colors
        
    def run(self, video_path=0, start_frame=0, conf_thresh=0.6):
        """ Runs the test on a video (or webcam)
        
        # Arguments
        video_path: A file path to a video to be tested on. Can also be a number, 
                    in which case the webcam with the same number (i.e. 0) is 
                    used instead
                    
        start_frame: The number of the first frame of the video to be processed
                     by the network. 
                     
        conf_thresh: Threshold of confidence. Any boxes with lower confidence 
                     are not visualized.
                    
        """
        
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
        
        vid_w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid_h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Skip frames until reaching start_frame
        if start_frame > 0:
            vid.set(cv2.CAP_PROP_POS_MSEC, start_frame)
            
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        
        input_size = self.input_shape[:2]
        
        while True:
            retval, img = vid.read()
            if not retval:
                print("Done!")
                return
                
            # model to predict 
            x = np.array([preprocess(img, input_size)])
            y = self.model.predict(x)
            
            result = self.prior_util.decode(y[0], confidence_threshold=conf_thresh)
            
            for r in result:
                xmin = int(round(r[0] * vid_w))
                ymin = int(round(r[1] * vid_h))
                xmax = int(round(r[2] * vid_w))
                ymax = int(round(r[3] * vid_h))
                conf = r[4]
                label = int(r[5])
                color = self.class_colors[label]
                text = self.class_names[label] + " " + ('%.2f' % conf)
                
                # draw box
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                
                # draw label
                text_top = (xmin, ymin-10)
                text_bot = (xmin + 90, ymin + 5)
                text_pos = (xmin + 5, ymin)
                cv2.rectangle(img, text_top, text_bot, color, -1)
                cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            
            # Calculate FPS
            # This computes FPS for everything, not just the model's execution 
            # which may or may not be what you want
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            
            # Draw FPS in top left corner
            cv2.rectangle(img, (0,0), (50, 17), (255,255,255), -1)
            cv2.putText(img, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            
            cv2.imshow("SSD detection", img)
            cv2.waitKey(10)


if __name__ == '__main__':

    from ssd_model import SSD300 as Model
    from ssd_utils import PriorUtil

    # Change this if you run with other classes than VOC
    class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    input_shape = (300,300,3)
    model = Model(input_shape, num_classes=len(class_names))
    prior_util = PriorUtil(model)

    # Change this path if you want to use your own trained weights
    model.load_weights('./models/ssd300_voc_weights_fixed.hdf5') 
            
    vid_test = VideoTest(model, prior_util, class_names, input_shape)
    
    # To test on webcam 0, remove the parameter (or change it to another number
    # to test on that webcam)
    try:
        #vid_test.run('data/video.mp4', start_frame=100)
        vid_test.run(0, conf_thresh=0.6)
    except KeyboardInterrupt:
        pass
    
    # python -m cProfile -o ssd_videotest.cprof ssd_videotest.py
    # pyprof2calltree -k -i ssd_videotest.cprof
    # kcachegrind ssd_videotest.cprof

