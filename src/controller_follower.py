import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist



class TurtlebotController(Node):

    def __init__(self):
        super().__init__('turtlebot_controller')
        self.subscription = self.create_subscription(
            Point,
            'centroid_data',
            self.centroid_callback,
            10)
        
        self.subscription  # prevent unused variable warning
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize variables for low-pass filter
        self.alpha = 0.2 # Smoothing factor orientation 
        self.beta = 0.5 # smoothing factor depth
        self.prev_x = None
        self.prev_z = None

        

        # Initialize PID controller gains and variables for linear velocity
        self.Kp_linear = 0.2
        self.Ki_linear = 0.0
        self.Kd_linear = 0.0
        self.integral_linear = 0.0
        self.previous_error_linear = 0.0

        
        # Initialize PID controller gains and variables for angular velocity
        self.Kp_angular = 0.0001
        self.Ki_angular = 0.0000001
        self.Kd_angular = 0.000001
        self.integral_angular = 0.0
        self.previous_error_angular = 0.0
        
        # Set reference values
        self.x_ref = 310
        self.z_ref = 3.0
        
    def centroid_callback(self, data):
        # Get the x coordinate of the centroid
        x = data.x
        z = data.z
        print(z)
        
        # Compute errors
        #x_error = self.x_ref - x
        #z_error = self.z_ref - z

        # Apply low-pass filter
        if self.prev_x is None:
            filtered_x = x
        else:
            filtered_x = self.alpha * x + (1 - self.alpha) * self.prev_x
        
        self.prev_x = filtered_x

        # Compute errors
        x_error = self.x_ref - filtered_x

        # Apply low-pass filter
        if self.prev_z is None:
            filtered_z = z
        else:
            filtered_z = self.beta * z + (1 - self.beta) * self.prev_z
        
        self.prev_z = filtered_z

        # Compute errors
        z_error = -self.z_ref + filtered_z
        
        
        # Compute proportional term for linear velocity
        
        P_linear = self.Kp_linear * z_error
        print(P_linear)


        # Compute integral term for linear velocity
        self.integral_linear += self.Ki_linear * z_error
        
        # Compute derivative term for linear velocity
        derivative_linear = self.Kd_linear * (z_error - self.previous_error_linear)
        self.previous_error_linear = z_error
        
        # Compute control output for linear velocity

        if z_error > 0:

           linear_vel = P_linear + self.integral_linear + derivative_linear

        else:

           linear_vel = 0
        
        # Compute proportional term for angular velocity
        P_angular = self.Kp_angular * x_error
        
        # Compute integral term for angular velocity
        self.integral_angular += self.Ki_angular * x_error
        
        # Compute derivative term for angular velocity
        derivative_angular = self.Kd_angular * (x_error - self.previous_error_angular)
        self.previous_error_angular = x_error
        
        # Compute control output for angular velocity
        angular_vel = P_angular + self.integral_angular + derivative_angular
        
        # Create the Twist message and publish it
        cmd_vel = Twist()
        cmd_vel.linear.x = float(linear_vel)
        cmd_vel.angular.z = float(angular_vel)
        # Print and plot the angular velocity
        print('Angular velocity:', angular_vel)
        print('linear velocity:', cmd_vel.linear.x)

        
        
        # Publish the Twist message
        self.publisher.publish(cmd_vel)

 


def main(args=None):
    rclpy.init(args=args)
    turtlebot_controller = TurtlebotController()
    rclpy.spin(turtlebot_controller)
    turtlebot_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
