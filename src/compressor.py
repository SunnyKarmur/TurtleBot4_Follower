import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


class ImageCompressor(Node):
    def __init__(self):
        super().__init__("image_compressor")
        self.bridge = CvBridge()

        # Create subscription to color image topic
        self.color_sub = self.create_subscription(
            Image,
            "/color/preview/image",
            self.color_callback,
            10
        )

        # Create subscription to depth image topic
        self.depth_sub = self.create_subscription(
            Image,
            "/stereo/depth",
            self.depth_callback,
            10
        )

        # Create publishers for compressed color image and compressed depth image
        self.color_pub = self.create_publisher(
            CompressedImage,
            "/camera/color/compressed",
            10
        )
        self.depth_pub = self.create_publisher(
            CompressedImage,
            "/camera/depth/compressed",
            10
        )

    def color_callback(self, msg):
        # Convert ROS2 Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Compress the image using JPEG format
        _, encoded_image = cv2.imencode(".jpg", cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 25])

        # Create a new CompressedImage message and publish it
        compressed_msg = CompressedImage()
        compressed_msg.format = "jpeg"
        compressed_msg.data = np.array(encoded_image).tostring()
        self.color_pub.publish(compressed_msg)

    def depth_callback(self, msg):
        # Convert ROS2 Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Compress the image using JPEG format
        _, encoded_image = cv2.imencode(".jpg", cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 25])

        # Create a new CompressedImage message and publish it
        compressed_msg = CompressedImage()
        compressed_msg.format = "jpeg"
        compressed_msg.data = np.array(encoded_image).tostring()
        self.depth_pub.publish(compressed_msg)


def main(args=None):
    rclpy.init(args=args)
    compressor = ImageCompressor()
    rclpy.spin(compressor)
    compressor.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
