""" A class for testing a SegLink model on a video file or webcam """

import numpy as np
import cv2
from timeit import default_timer as timer

from ssd_data import preprocess
from sl_utils import rbox_to_polygon


class VideoTest(object):
    """ Class for testing a trained SSD model on a video file and show the
        result in a window. Class is designed so that one VideoTest object 
        can be created for a model, and the same object can then be used on 
        multiple videos and webcams.
        
        Arguments:
            model:       An SSD model. It should already be trained for 
                         images similar to the video to test on.
                         
            input_shape: The shape that the model expects for its input, 
                         as a tuple, for example (300, 300, 3)    
                         
            bbox_util:   An instance of the BBoxUtility class in ssd_utils.py
                         The BBoxUtility needs to be instantiated with 
                         the same number of classes as the length of        
                         class_names.
    
    """
    
    def __init__(self, model, prior_util, input_shape):
        self.model = model
        self.input_shape = input_shape
        self.prior_util = prior_util
        
    def run(self, video_path=0, start_frame=0, segment_threshold=0.55, link_threshold=0.45):
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
        
        # skip frames until reaching start_frame
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
            
            result = self.prior_util.decode(y[0], segment_threshold, link_threshold)
            
            for r in result:
                xy = rbox_to_polygon(r[:5])
                xy = xy / input_size * [vid_w, vid_h]
                xy = xy.reshape((-1,1,2))
                xy = np.round(xy)
                xy = xy.astype(np.int32)
                cv2.polylines(img, [xy], True, (0,0,255))
                
            # calculate fps
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            
            # draw fps
            cv2.rectangle(img, (0,0), (50, 17), (255,255,255), -1)
            cv2.putText(img, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            
            cv2.imshow("SegLink detection", img)
            cv2.waitKey(10)


if __name__ == '__main__':

    from sl_model import DSODSL512 as Model
    from sl_utils import PriorUtil

    input_shape = (512,512,3)
    model = Model(input_shape)
    prior_util = PriorUtil(model)

    model.load_weights('./checkpoints/201711132011_dsodsl512_synthtext/weights.001.h5', by_name=True)
    
    vid_test = VideoTest(model, prior_util, input_shape)

    # To test on webcam 0, /dev/video0
    try:
        #vid_test.run('data/video.mp4', start_frame=100)
        vid_test.run(video_path=0, start_frame=0, segment_threshold=0.55, link_threshold=0.45)
    except KeyboardInterrupt:
        pass
    
