import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge

class DepthImageSubscriber(Node):

    def __init__(self):
        super().__init__("depth_image_subscriber")
        self.subscription = self.create_subscription(
            Image,
            "/stereo/depth",
            self.process_image,
            10
        )
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
    def process_image(self, msg):
        # Convert ROS Image message to OpenCV image
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Get the depth value of the desired pixel
        x = 320  # example x coordinate
        y = 240  # example y coordinate
        depth_value = depth_image[y, x]  # assuming 8-bit depth map image

    # Convert the depth value to meters
        depth_meters = depth_value / 1000.0  # assuming depth map values are in millimeters
        # Display the depth image
        #cv2.imshow("Depth Image", depth_image)
        #cv2.waitKey(1)  # needed to show the image

        # Print the distance of the desired pixel
        print(f"Depth at ({x}, {y}): {depth_meters} m")

def main(args=None):
    rclpy.init(args=args)
    depth_image_subscriber = DepthImageSubscriber()
    rclpy.spin(depth_image_subscriber)
    depth_image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
