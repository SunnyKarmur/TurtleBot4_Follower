#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from geometry_msgs.msg import Point

class PedestrianTracker(Node):
    def __init__(self):
        super().__init__('pedestrian_tracker')
        self.bridge = CvBridge()
        self.trackers = []
        self.pub = self.create_publisher(Point, '/follow_object', 10)
        self.subscription = self.create_subscription(Image, '/pedestrian_detection/image_raw', self.image_callback, 10)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Detect and track pedestrians using HOG descriptors and the MOSSE tracker
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(cv_image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        new_trackers = []
        for bbox in rects:
            x, y, w, h = bbox
            tracker = cv2.legacy.TrackerMOSSE.create()
            ok = tracker.init(cv_image, bbox)
            if ok:
                new_trackers.append(tracker)
                center_x = x + w/2
                center_y = y + h/2
                pos_msg = Point()
                pos_msg.x = center_x
                pos_msg.y = center_y
                pos_msg.z = 0.0  # Ensure z is a float
                self.pub.publish(pos_msg)

        self.trackers = new_trackers

def main(args=None):
    rclpy.init(args=args)

    tracker = PedestrianTracker()

    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass

    tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
