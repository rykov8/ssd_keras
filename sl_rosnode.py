""" A class for performing SegLink detetion on ROS Image topics """

import numpy as np
import cv2
from timeit import default_timer as timer

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ssd_data import preprocess
from sl_utils import rbox_to_polygon

import tensorflow as tf


class RosVideoTest:
    """ Class for performing SegLink detection on a ROS Image topic and 
    publishing the results on a secod Image topic.
    
    Arguments:
        model: Trained SegLink model.
        prior_util: PriorUtil object for decoding the output of the SegLink 
            model.
        class_names: List of strings, each containing the name of a class.
            The first name should be that of the background class
            which is not used.
        input_shape: The shape that the model expects for its input, 
            as a tuple, for example (512, 512, 3),
    """
    def __init__(self, model, prior_util, class_names, input_shape):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = model
        self.input_shape = input_shape
        self.prior_util = prior_util
        
        self.graph = tf.get_default_graph()
        self.prev_time = timer()
        
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        
    def callback(self,data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        input_size = self.input_shape[:2]
        
        vid_h, vid_w = img.shape[:2]
        
        # model to predict 
        x = np.array([preprocess(img, input_size)])
        
        with self.graph.as_default():
            y = self.model.predict(x)
        
        result = self.prior_util.decode(y[0], segment_threshold=0.55, link_threshold=0.45)
        
        for r in result:
            xy = rbox_to_polygon(r[:5])
            xy = xy / input_size * [vid_w, vid_h]
            xy = xy.reshape((-1,1,2))
            xy = np.round(xy)
            xy = xy.astype(np.int32)
            cv2.polylines(img, [xy], True, (0,0,255))
            
        # calculate fps
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        accum_time = self.accum_time = self.accum_time + exec_time
        self.curr_fps = self.curr_fps + 1
        if accum_time > 1:
            accum_time = self.accum_time = accum_time - 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0
        
        # draw fps
        cv2.rectangle(img, (0,0), (50, 17), (255,255,255), -1)
        cv2.putText(img, self.fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        
        #cv2.imshow("SegLink detection", img)
        #cv2.waitKey(10)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)
    
    def start(self, video_path=0, start_frame=0, segment_threshold=0.9, link_threshold=0.7):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/cam", Image, self.callback)
        self.image_pub = rospy.Publisher("/image_topic", Image)


if __name__ == '__main__':

    from sl_model import SL512 as Model
    from sl_utils import PriorUtil
    from ssd_utils import load_weights

    class_names = ['Background', 'Text'];
    input_shape = (512,512,3)
    model = Model(input_shape)
    prior_util = PriorUtil(model)

    load_weights(model, './checkpoints/201711132011_dsodsl512_synthtext/weights.001.h5')
    
    ros_vid_test = RosVideoTest(model, prior_util, class_names, input_shape)
    rospy.init_node('scene_text_reader', anonymous=True)
    try:
        ros_vid_test.start()
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()



## master
# roscore -p 11311
# export ROS_MASTER_URI=http://172.21.12.239:11311/
# export ROS_IP=172.21.12.239

## client
# export ROS_MASTER_URI=http://172.21.12.239:11311/
# export ROS_IP=172.21.6.194
