#!/usr/bin/env python
from __future__ import print_function

import rospy
import cv2
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


if __name__ == '__main__':
    
    node_name = 'image_publisher'
    video_path = '/dev/video0'
    topic_name = '/cam'
    visualize = False
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        topic_name = sys.argv[2]
    
    bridge = CvBridge()
    vid = cv2.VideoCapture(video_path)
    
    image_pub = rospy.Publisher(topic_name, Image, queue_size=1)
    rospy.init_node(node_name, anonymous=True)
    
    print('publish %s at %s' % (video_path, topic_name))
    
    while not rospy.is_shutdown():
        retval, img = vid.read()
        if not retval:
            print("Done!")
            break
        
        if visualize:
            cv2.imshow("Image window", img)
            cv2.waitKey(3)

        try:
            image_pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)
    
    cv2.destroyAllWindows()
