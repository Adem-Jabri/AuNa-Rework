import rclpy
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from auna_rl.robot_controller import robotController
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from rclpy.node import Node
from gazebo_msgs.srv import  SetEntityState
from std_srvs.srv import Empty
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Point, Quaternion
import random
import time

class aunaEnvironment(robotController, Env):
    def __init__(self, size=100):
        super().__init__()
        robotController.__init__(self)
        self.size = size
        
        # initialize the robot parameters: start position, target position
        self._target_locations = [
            (0, -0.35, 0),              # Point 1  
            (11.2, 2.4, 0.01),          # corner 2  
            (10.2, 14.6, 0.01),         # corner 3   
            (-0.68, 15.66, 0.01),       # corner 4  
            (-2.2, 12.318, 0.01),       # corner 5  
            #(4.233, 5.517, 0.01),       # Point 6 (Middle of the semicircle)  
            (-3, 5.3, 0.01),            # Point 7 (Start of the straight line)  
            #(-12.359, 14.657, 0.01),    # Point 8 (End of the straight line)   
            (-18.1, 13.879, 0.01),      # Point 9 (End of the second semicircle)
            (-14.657, 1.2, 0.01)        # Point 10 (Last Corner)
        ]
        self._agent_locations = self._target_locations.copy()
        
         # Possible orientations (x, y, z, w). For each position it will be 2 orientations 
        self.orientations = [
            ((0, 0, 0, 1), (0, 0, 1, 0)),  
            ((0, 0, 1, 1.7), (0, 0, 1, -0.5)),
            ((0, 0, 1, 0.4), (0, 0, 1, -2)),
            ((0, 0, 1, 0), (0, 0, 0, 1)),
            ((0, 0, 1, -1.9), (0, 0, 1, 1)),
            #((0, 0, 1, -0.5), (0, 0, 1, 2)),
            ((0, 0, 1, 0.35), (0, 0, 1, -2.2)),
            #((0, 0, 1, 0.35), (0, 0, 1, -2.2)),
            ((0, 0, 1, -0.75), (0, 0, 1, 1.4)),
            ((0, 0, 1, -5), (0, 0, 1, 0.25)),
        ]

        self._minimum_distance_from_obstacles = 0.25
        self._minimum_distance_from_target = 0.6
        self._best_distance_from_obstacles = 0.5
        self.remaining_waypoints = []
        self.original_distance_to_target = 0
        self.info_received = False
        self.create_ros_clients()
        self.collision_number = 0
        self.current_velocity = 0 
        self.last_time = self.get_clock().now().to_msg().sec
        self.angular_velocity = 0
        self.current_velocity = 0
        self.vel_sent = False 
        self.target_reached_reward = 0

        all_locations = np.array(self._target_locations)
        min_bounds = np.min(all_locations, axis=0) 
        max_bounds = np.max(all_locations, axis=0) + 0.001
        
        self.action_space = spaces.Box(low=np.array([0.5, -0.5]), high=np.array([1.0, 0.5]), dtype=np.float32)  # Continuous action space
        self.observation_space = spaces.Dict({
            "lidar_readings": spaces.Box(low=0.2, high=30, shape=(10,), dtype=np.float32),
            "velocity": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "steering_angle": spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        })
    
    def step(self, action):
        
        terminated = False
        
        linear_vel, angular_vel = action
        self.send_vel(linear_vel, angular_vel)

        self.spin()

        # We use `np.clip` to make sure we don't leave the grid
        #self._agent_location = np.clip(
        #    self._agent_location + direction, 0, self.size - 1
        #)
        
        observation = self._get_obs()
        info = self._get_info(False)
        
        print(f"observation:{observation}")
        while self.info_received == False:
            rclpy.spin_once(self)

        if (self.actual_position.z > 0.1) :
            terminated = True
            self.reset()
        #print (f"++++++++++++++++++++++++++{self._target_location}++++++++++++++++++{self.actual_position}")
        #print (f"---------------{self.remaining_waypoints}-----------------------")
        #print(f"+++++++++++++++++{info}+++++++++++++++++++++")
        
        # covered distance calculation
        current_time = self.get_clock().now().to_msg().sec
        time_elapsed = current_time - self.last_time
        self.last_time = self.get_clock().now().to_msg().sec
        #distance_covered = self.current_velocity * time_elapsed
        #self.total_distance_covered += distance_covered

        

        # reward calculation
        collision_calculation = self.calculate_collision_reward(self.distance_to_the_closest_obstacle(info["laser"]))
        
            
        #if (self.check_obstacle_proximity(info["laser"])) : 
        #    terminated = True
        #    collision_penalty = -50
        #    self.get_logger().info("The robot hits an obstacle")

        if (collision_calculation == -70) : 
            terminated = True
        
        self.target_reached_reward = 100 if (info["distance"] < self._minimum_distance_from_target) else 0
        if (self.target_reached_reward > 0) : 
            terminated = True
            self.get_logger().info("The target is reached")

        # progress reward based on cumulative distance
        if (collision_calculation > 0):
            if (self.original_distance_to_target > 0):
                progress_reward = (abs(self.original_distance_to_target - info["distance"]) / self.original_distance_to_target ) * 10
        else: progress_reward = 0

        # steering punishment to discourage excessive steering
        steering_penalty = -abs(angular_vel) * 2
        
        #move_in_the_middle_reward = 1 if (self.distance_from_closest_waypoint_to_target > 0.4) else (if (self.distance_from_closest_waypoint_to_target))

        # encourages speed
        time_penalty = -1 
        
        reward = progress_reward + collision_calculation + time_penalty + self.target_reached_reward + steering_penalty
        remain = info["distance"]
        print(f"original: {self.original_distance_to_target}+++++++++++actual: remaining{remain}")
        print(f"progress_rewrd : {progress_reward}+++++collision_calculation: {collision_calculation}")
        #print(f"time elapsed : {time_elapsed}++++++++++++++++++time_penalty: {time_penalty}")
        #print(f"steering_penalty: {steering_penalty}+++++++++")
        #print(f"calculation: {collision_calculation}")
        return observation, reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Reset the simulation world 
        future = self.reset_world_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        self.random_position = random.choice([5, 6])
        #self.random_position = 0

        #self.closest_waypoint_index = 0
        # Select a random start position and orientation
        self._agent_location = np.array(self._agent_locations[self.random_position], dtype=np.float32)
        # 0 for forward, 1 for backward
        choices = [0, 1]
        weights = [1, 2]
        self.random_orientation = random.choices(choices, weights, k=1)[0]
        #self.random_orientation = 1
        orientation = self.orientations[self.random_position][self.random_orientation]
        # Set the robot to a new state with orientation, 
        self.set_robot_state(self._agent_location, orientation)  
        
    
        # Boolean variable that waits until the set_robot_state method finishes.
        while self.robot_reset_done == False:
            rclpy.spin_once(self)

        # Sample the target's location randomly until it does not coincide with the agent's location
        #self._target_location = self._agent_location
        #while np.array_equal(self._target_location, self._agent_location):
        #    self.target_index = self._agent_locations.index(random.choice(self._agent_locations))
        #    self.x = self._agent_locations[self.target_index]
        #    self._target_location = np.array(self._agent_locations[self.target_index], dtype=np.float32)

        # determine the target location based on the direction, it will be the waypoint 
        # directly after the start point.
        if self.random_orientation == 0:  # Forward
            self.target_index = (self.random_position + 1) % len(self._agent_locations)
        else:  # Backward
            self.target_index = (self.random_position - 1) % len(self._agent_locations)
        self._target_location = np.array(self._agent_locations[self.target_index], dtype=np.float32)

        self.spin()

        self.last_time = self.get_clock().now().to_msg().sec

        # Create observation and info dictionary
        observation = self._get_obs()
        info = self._get_info(True)
        
        while self.info_received == False:
            rclpy.spin_once(self)

        return observation, info

    
    def render(self): pass

    def close(self): pass
    
    def check_obstacle_proximity(self, laser_readings):
        #Check if any obstacles are too close based on laser readings.
        for distance in laser_readings:
            if distance < self._minimum_distance_from_obstacles:
                return True
        return False

    def distance_to_the_closest_obstacle(self, laser_readings):
        #Check if any obstacles are too close based on laser readings.
        min_dist = 29
        for distance in laser_readings:
            if distance < min_dist:
                min_dist = distance
        return min_dist    

    def calculate_collision_reward(self, distance):
        if distance > 0.4:
            return 5
        elif distance > 0.3:
            return 3
        elif distance > 0.21:
            return -10
        else:
            self.collision_number += 1
            return -70

    def check_waypoint_reached(self):
        reached = False
        if (calculate_distance(self.remaining_waypoints[0], self.actual_position) <= 0.1):
            reached = True 
        #print(f"+++++++++++{self.remaining_waypoints[0]}+++++++++++++{self.actual_position}")
        return reached 

    def spin(self):
        # the node will be spun untill the sensor readings are recieived 
        self.readings_received = False
        self.odom_received = False
        while ((self.readings_received == False) & (self.odom_received == False)):
            rclpy.spin_once(self)

    def create_ros_clients(self):
        # Initialize ROS2 service clients
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        self.set_entity_state_client = self.create_client(SetEntityState, '/set_entity_state')
        # Wait for the services to be available
        self.reset_world_client.wait_for_service()
        self.set_entity_state_client.wait_for_service()

    def reset_simulation_world(self):
        # Call the reset_world service
        future = self.reset_world_client.call_async(Empty.Request())
        # Add a timeout of 5 seconds
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)  
        
        if future.done():
            if future.result() is not None:
                self.get_logger().info('World reset successfully')
            else:
                self.get_logger().error('Failed to call service reset_world')
        else:
            self.get_logger().error('Service call reset_world did not finish before timeout')
    
    def set_robot_state(self, position, orientation):
        # Construct the request to set entity state
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = 'robot'  
        req.state.pose = Pose(
            position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
            orientation=Quaternion(x=float(orientation[0]), y=float(orientation[1]), z=float(orientation[2]), w=float(orientation[3]))
        )
        req.state.twist.linear.x = float(0)
        req.state.twist.linear.y = float(0)
        req.state.twist.linear.z = float(0)
        req.state.twist.angular.x = float(0)
        req.state.twist.angular.y = float(0)
        req.state.twist.angular.z = float(0)

        # Call the set_entity_state service
        future = self.set_entity_state_client.call_async(req)
        # Add a timeout of 5 seconds
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)  

        if future.done():
            if future.result().success:
                self.get_logger().info('Robot state set successfully') 
            else:
                self.get_logger().error('Failed to set robot state')
        else:
            self.get_logger().error('Service call set_entity_state did not finish before timeout')
        self.robot_reset_done = True   


    def _get_obs(self):
        return {
        "lidar_readings": np.array(self.readings, dtype=np.float32), 
        "velocity": np.array([self.current_velocity], dtype=np.float32),  
        "steering_angle": np.array([self.angular_velocity], dtype=np.float32)  
        }

    def _get_info(self, reset):
        agent_location = np.array(self.actual_position)
        target_location = np.array(self._target_location)
        distance = self.calculate_distance_to_the_target(reset)
        done = self.target_reached_reward > 0
        return {"distance": distance, "laser": self.readings, "done": done}

    def calculate_distance(self, point1, point2):
        #distance = np.linalg.norm(np.array(np.array(point2, dtype=np.float32)) - np.array(np.array(point1, dtype=np.float32)))
        #print (f"{point1}*************{point2}")
        #distance = np.linalg.norm(point2 - point1)
        point1 = np.array([point1.x, point1.y, point1.z]) if isinstance(point1, Point) else np.array(point1)
        point2 = np.array([point2.x, point2.y, point2.z]) if isinstance(point2, Point) else np.array(point2)
        distance = np.linalg.norm(point2 - point1)
        return distance

    def sum_pairwise_distances(self, waypoints):
        total_distance = 0
        for i in range(len(waypoints) - 1):
            total_distance += self.calculate_distance(waypoints[i], waypoints[i+1])
        
        return total_distance

    def calculate_distance_to_the_target(self, reset):
        if (reset):
            if (self.random_orientation == 1):
                agent_locations = self._agent_locations[::-1]
                real_start_index = len(self._agent_locations) - self.random_position - 1
                real_target_index = len(self._agent_locations) - self.target_index - 1
                if real_start_index <= real_target_index:
                    self.remaining_waypoints = agent_locations[real_start_index + 1:real_target_index + 1]
                else:
                    if (real_start_index == len(self._agent_locations) - 1):
                        self.remaining_waypoints = agent_locations[:real_target_index + 1]
                    else:
                        self.remaining_waypoints = agent_locations[real_start_index + 1:] + agent_locations[:real_target_index + 1]
            else:
                agent_locations = self._agent_locations 
                real_start_index = self.random_position
                real_target_index = self.target_index
                if real_start_index <= real_target_index:
                    self.remaining_waypoints = agent_locations[real_start_index + 1:real_target_index + 1]
                else:
                    if (real_start_index == len(self._agent_locations) - 1):
                        self.remaining_waypoints = agent_locations[:real_target_index + 1]
                    else:
                        self.remaining_waypoints = agent_locations[real_start_index + 1:] + agent_locations[:real_target_index + 1]
            
        if (len(self.remaining_waypoints) >= 2):    
            distance_from_closest_waypoint_to_target = self.sum_pairwise_distances(self.remaining_waypoints)
            distance_to_the_closeset_point = self.calculate_distance(self.remaining_waypoints[0], self.actual_position)
            self.info_received = True
            if (reset): self.original_distance_to_target = distance_to_the_closeset_point + distance_from_closest_waypoint_to_target
            return distance_to_the_closeset_point + distance_from_closest_waypoint_to_target
        elif (len(self.remaining_waypoints) >= 1):
            distance_to_the_closeset_point = self.calculate_distance(self.remaining_waypoints[0], self.actual_position)
            self.info_received = True
            if (reset): self.original_distance_to_target = distance_to_the_closeset_point
            return distance_to_the_closeset_point
        else:
            self.info_received = True
            #self.get_logger().info("---------------------0000000000000000000-----------------------") 
            return 0
        

    def run_PPO(self):
        # checks the environment and outputs additional warnings if needed    
        check_env(self)
        self.get_logger().info("check finished")
        
        model = PPO("MultiInputPolicy", self, verbose=1)
        
        model.learn(total_timesteps=800000)
        print(f"learning is finished")
        
        # Enjoy trained agent, and test it 
        obs, info = self.reset()
        print("first reset done --------------------------------------------")
        episode_reward = 0
        true_positives = 0
        test_episodes = 50000
        for _ in range(test_episodes):
            action, _states = model.predict(obs)
            # predict() gets the model’s action from an observation
            obs, reward, done, truncated, info = self.step(action)
            print(f"------------------------{done}{done}{done}{done}{done}{done}{done}{done}------------------------------------")
            episode_reward += reward
            # I think that this should be fixed because 'done' will now be set to True 
            # if the robot either terminates or hits an obstacle.
            if done : 
                true_positives += 1
                obs, info = self.reset()
                ("second reset done ++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"i am in the loop*************************************************************************************")    
        acc = true_positives / test_episodes
        print(f"Accuracy: {acc*100}%")
        model.save("~/AuNa-Rework/packages/src/auna_rl")

def main(args = None):
        rclpy.init()
        auna = aunaEnvironment()
        rclpy.spin(auna)
        
if __name__=="__main__": 
        main()    