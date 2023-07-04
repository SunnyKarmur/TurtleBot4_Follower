import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class PedestrianDetector(Node):

    def __init__(self):
        super().__init__('pedestrian_detector')
        self.bridge = CvBridge()
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = self.tracker_types[6]  # Select MOSSE tracker
        self.trackers = []
        self.subscription = self.create_subscription(
            Image,
            '/color/preview/image',
            self.callback,
            10)
        self.publisher_ = self.create_publisher(Image, '/pedestrian_detection/image_raw', 10)

    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        pedestrians, _ = self.hog.detectMultiScale(cv_image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        new_trackers = []
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

def main(args=None):
    rclpy.init(args=args)
    node = PedestrianDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
