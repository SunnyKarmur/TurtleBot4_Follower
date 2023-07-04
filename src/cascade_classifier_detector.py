import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CascadeDetectorNode(Node):
    def __init__(self):
        super().__init__('cascade_detector_node')
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("Cascade Classifier", image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    cascade_detector_node = CascadeDetectorNode()
    rclpy.spin(cascade_detector_node)
    cascade_detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()