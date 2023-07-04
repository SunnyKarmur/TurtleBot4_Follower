#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from geometry_msgs.msg import Point

class PedestrianDetectorTracker(Node):

    def __init__(self):
        super().__init__('pedestrian_detector_tracker')
        self.bridge = CvBridge()
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = self.tracker_types[6]  # Select MOSSE tracker
        self.trackers = []
        self.publisher_ = self.create_publisher(Image, '/pedestrian_detection/image_raw', 10)
        self.pub = self.create_publisher(Point, '/follow_object', 10)
        self.subscription = self.create_subscription(Image, '/image_raw', self.callback, 10)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        pedestrians, _ = self.hog.detectMultiScale(cv_image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        new_trackers = []
        center_x = 0
        center_y = 0
        for (x, y, w, h) in pedestrians:
            found = False
            for tracker in self.trackers:
                ok, bbox = tracker.update(cv_image)
                if ok:
                    x, y, w, h = bbox
                    cv2.rectangle(cv_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                    found = True
                    new_trackers.append(cv2.legacy.TrackerMOSSE_create())
                    new_trackers[-1].init(cv_image, tuple(map(int, bbox)))
                    break
            if not found:
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                new_trackers.append(cv2.legacy.TrackerMOSSE_create())
                new_trackers[-1].init(cv_image, (x, y, w, h))
        self.trackers += new_trackers

        processed_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.publisher_.publish(processed_image)

        # Publish the position of the object
        if self.trackers:
            ok, bbox = self.trackers[0].update(cv_image)
            if ok:
               x, y, w, h = [int(v) for v in bbox]
            center_x = x + w/2
            center_y = y + h/2
            pos_msg = Point()
            pos_msg.x = center_x
            pos_msg.y = center_y
            pos_msg.z = 0.0
            self.pub.publish(pos_msg)
def main(args=None):
    rclpy.init(args=args)
    node = PedestrianDetectorTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
