from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import rclpy

class robotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.num_lidar_sections = 10
        self.readings = []
        # publish Actions to control the velocity of the robot
        self.action_publisher = self.create_publisher(Twist, 'robot/cmd_vel', 10)
        # subscribe to position and Lidar readings, to know the state of the robot
        self.lidar_subscriber = self.create_subscription(LaserScan, 'robot/scan', self.sub_lidar_callback, qos_profile=qos_profile_sensor_data)
    
    def send_vel(self, linear, angular):
        print(f"Sending velocity - Linear: {linear}, Angular: {angular}")
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.action_publisher.publish(msg)
    
    def sub_lidar_callback(self, msg: LaserScan):
        print(f"Lidar callback triggered with {len(msg.ranges)} ranges.")
        # dealing with only 10 rays, which are the min of each section 
        #(the readings will be sampled to 10 sections)
        self.min_per_section = []
        self.readings = np.array(msg.ranges)
        self.readings_per_section = len(self.readings) // self.num_lidar_sections
        for i in range(self.num_lidar_sections):
            start_index = i * self.readings_per_section
            end_index = (i+1) * self.readings_per_section
            # Ensure the index does not exceed the length of readings array
            end_index = min(end_index, len(self.readings))
            self.min_per_section.append(min(self.readings[start_index:end_index]))
        self.readings = self.min_per_section
        self.received = True 
def main(args = None):
        rclpy.init()

        controller = robotController()

        rclpy.spin(controller)
        
if __name__=="__main__": 
        main()    