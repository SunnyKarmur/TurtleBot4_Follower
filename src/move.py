import rclpy
from geometry_msgs.msg import Twist

def move_turtlebot4(linear_speed, angular_speed):
    # Initialize ROS node
    rclpy.init()
    node = rclpy.create_node('turtlebot4_controller')

    # Create a publisher for the Twist message
    publisher = node.create_publisher(Twist, '/cmd_vel', 10)

    # Create a Twist message
    twist = Twist()
    twist.linear.x = linear_speed
    twist.angular.z = angular_speed

    # Publish the Twist message
    publisher.publish(twist)

    # Sleep for a short time to allow the message to be published
    node.get_logger().info("Moving turtlebot4 with linear speed: {}, angular speed: {}".format(linear_speed, angular_speed))
    rclpy.spin_once(node, timeout_sec=0.1)

    # Shutdown the ROS node
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # Move turtlebot4 with linear speed of 0.2 m/s and angular speed of 0.4 rad/s
    move_turtlebot4(0.2, 0.4)
