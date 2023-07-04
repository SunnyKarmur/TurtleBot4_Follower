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
        self.tracker = None
        self.pub = self.create_publisher(Point, '/follow_object', 10)
        self.subscription = self.create_subscription(Image, '/pedestrian_detection/image_raw', self.image_callback, 10)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Your detection and tracking code here

        # Publish the position of the object
        if self.tracker is not None:
            ok, bbox = self.tracker.update(cv_image)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                center_x = x + w/2
                center_y = y + h/2
                pos_msg = Point()
                pos_msg.x = center_x
                pos_msg.y = center_y
                pos_msg.z = 0
                self.pub.publish(pos_msg)


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

