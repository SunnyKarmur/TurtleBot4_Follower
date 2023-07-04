import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class SsdDetectorNode(Node):
    def __init__(self):
        super().__init__('ssd_detector_node')
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.net = cv2.dnn.readNetFromCaffe('path/to/prototxt', 'path/to/caffemodel')

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.imshow("SSD Detector", image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    ssd_detector_node = SsdDetectorNode()
    rclpy.spin(ssd_detector_node)
    ssd_detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
