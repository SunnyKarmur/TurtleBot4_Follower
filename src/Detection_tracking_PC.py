"""

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

class PersonTracker(Node):
    def __init__(self):
        super().__init__('person_tracker')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.image_pub = self.create_publisher(Image, '/person_tracking/image', 10)
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.tracker = cv2.TrackerCSRT_create()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().info(str(e))
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            bbox = (x, y, w, h)
            ok = self.tracker.init(cv_image, bbox)

        if not self.tracker:
            self.get_logger().warn('Tracker not initialized')
            return

        ok, bbox = self.tracker.update(cv_image)

        if ok:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    tracker = PersonTracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

"""
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

class PersonTracker(Node):
    def __init__(self):
        super().__init__('person_tracker')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.image_pub = self.create_publisher(Image, '/person_tracking/image', 10)
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.tracker = cv2.legacy.TrackerMOSSE_create()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().info(str(e))
            return

        rects, _ = self.detector.detectMultiScale(cv_image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        if len(rects) > 0:
            x, y, w, h = rects[0]
            bbox = (x, y, w, h)
            ok = self.tracker.init(cv_image, bbox)

        if not self.tracker:
            self.get_logger().warn('Tracker not initialized')
            return

        ok, bbox = self.tracker.update(cv_image)

        if ok:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    tracker = PersonTracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
