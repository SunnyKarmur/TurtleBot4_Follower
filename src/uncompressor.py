import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_uncompressor")
        self.bridge = CvBridge()
        self.color_subscriber = self.create_subscription(
            CompressedImage,
            "/camera/color/compressed",
            self.color_callback,
            10
        )
        self.depth_subscriber = self.create_subscription(
            CompressedImage,
            "/camera/depth/compressed",
            self.depth_callback,
            10
        )
        self.color_publisher = self.create_publisher(
            Image,
            "/camera/color/image",
            10
        )
        self.depth_publisher = self.create_publisher(
            Image,
            "/camera/depth/image",
            10
        )

    def color_callback(self, msg):
        # Convert the compressed image data to a numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)

        # Decode the compressed image using OpenCV
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert the OpenCV image to a ROS2 Image message
        ros2_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # Set the message header
        ros2_msg.header.stamp = self.get_clock().now().to_msg()

        # Publish the uncompressed color image to the /camera/color/image topic
        self.color_publisher.publish(ros2_msg)

    def depth_callback(self, msg):
        # Convert the compressed image data to a numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)

        # Decode the compressed image using OpenCV
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        # Convert the 16-bit depth image to an 8-bit grayscale image
        cv_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # Convert the OpenCV image to a ROS2 Image message
        ros2_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="mono8")

        # Set the message header
        ros2_msg.header.stamp = self.get_clock().now().to_msg()

        # Publish the uncompressed depth image to the /camera/depth/image topic
        self.depth_publisher.publish(ros2_msg)


def main(args=None):
    rclpy.init(args=args)
    subscriber = ImageSubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
