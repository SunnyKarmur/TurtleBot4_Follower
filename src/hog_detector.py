import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class HogDetectorNode(Node):
    def __init__(self):
        super().__init__('hog_detector_node')
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects, _ = self.hog.detectMultiScale(gray_image, winStride=(8, 8), padding=(32, 32), scale=1.05)
        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("HOG Detector", image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    hog_detector_node = HogDetectorNode()
    rclpy.spin(hog_detector_node)
    hog_detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
