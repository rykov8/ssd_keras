#!/usr/bin/env python
from __future__ import print_function

import rospy
import cv2
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


if __name__ == '__main__':
    
    node_name = 'image_listener'
    topic_name = '/cam'
    
    if len(sys.argv) > 1:
        topic_name = sys.argv[1]
    
    bridge = CvBridge()
    
    def callback(data):
        try:
            img = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image window", img)
        cv2.waitKey(3)
    
    image_sub = rospy.Subscriber(topic_name, Image, callback)
    rospy.init_node(node_name, anonymous=True)
    
    print('listen on %s' % (topic_name,))
    
    try:
        rospy.spin()    
    except KeyboardInterrupt:
        print("Shutting down")
    
    cv2.destroyAllWindows()
