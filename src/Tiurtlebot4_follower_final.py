#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist

class ObjectFollower(Node):

    def __init__(self):
        super().__init__('object_follower')
        self.subscription = self.create_subscription(Point, '/follow_object', self.callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_x = 0.0
        self.target_y = 0.0

    def callback(self, data):
        self.target_x = data.x
        self.target_y = data.y

    def run(self):
        while rclpy.ok():
            # Calculate control signals
            twist_msg = Twist()
            twist_msg.linear.x = 0.1  # move forward
            twist_msg.angular.z = 0.0

            if self.target_x != 0.0:
                # Calculate error signal
                error_x = self.target_x - 320  # assume camera image is 640x480
                twist_msg.angular.z = -error_x / 1000  # proportional control

            # Publish control signals
            self.publisher_.publish(twist_msg)
            self.get_logger().info('Publishing Twist message: linear.x=%f, angular.z=%f' % (twist_msg.linear.x, twist_msg.angular.z))
            rclpy.spin_once(self)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectFollower()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
