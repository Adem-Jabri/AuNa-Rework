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
            #(9.626048021284198, 0.9948578832933341, 0.009999696821740872),  # Ecke 1
            #(9.626048021284198, 0.9948578832933341, 0.009999696821740872),  # Ecke 2
            (-0.3, -0.3, 0.009999696821740872),  # Ecke 3
            #(-1.9054505190339759, 12.31810289360229, 0.009999582055632297),  # Ecke 4
            #(2.8681651239680486, 11.467601836868635, 0.009999680398369362),  # Ecke 5
            #(4.233035419730203, 5.517313709660946, 0.009999545338471208),    # Point 6 (Mitte Halbkreis)
            #(-1.5898110268184236, 4.662859418442022, 0.009999649799570381),  # Point 7 (Anfang Gerade)
            #(-12.35910483131254, 14.657580958662619, 0.009999443490755253),  # Point 8 (Ende Gerade)
            #(-17.934198543818844, 13.879836779953955, 0.009995890946504368), # Point 9 (Ende zweite Halbes Kreis)
            (-14.657546342770011, 1.0, 0.00999940416938419)   # Point 10 (Letzte Ecke)
        ]
        self._agent_locations = self._target_locations.copy()

        self._minimum_distance_from_obstacles = 0.25
        self._minimum_distance_from_target = 0.2
         # Possible orientations (x, y, z, w)
        self.orientations = [
            (0.0, 0.0, 1.0, 0.0),  # 180 degree rotation around Z-axis
            (0.0, 0.0, 0.0, 1.0),  # No rotation
        ]
        
        self.create_ros_clients()

        all_locations = np.array(self._target_locations)
        min_bounds = np.min(all_locations, axis=0)
        max_bounds = np.max(all_locations, axis=0)
        
        self.action_space = spaces.Discrete(3)
        """
        initialize size, in the gym Doc, the size of the square grid was initialized

        Observations are dictionaries with the agent's and the target's location.
        Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        """
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=min_bounds, high=max_bounds, shape=(3,), dtype=np.float32),
            "target": spaces.Box(low=min_bounds, high=max_bounds, shape=(3,), dtype=np.float32),
        })
    
    def step(self, action):
        
        terminated = False
        
        directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])]
        direction = directions[action]
        linear_vel = 1.0
        angular_vel = -0.5 if action == 0 else 0.5 if action == 2 else 0.0

        self.send_vel(linear_vel, angular_vel)

        self.spin()

        # We use `np.clip` to make sure we don't leave the grid
        #self._agent_location = np.clip(
        #    self._agent_location + direction, 0, self.size - 1
        #)
        
        observation = self._get_obs()
        info = self._get_info()
        
        # reward calculation
        if (info["distance"] < self._minimum_distance_from_target):
            # If the agent reached the target it gets a positive reward
            reward = 1
            terminated = True
            self.get_logger().info("The target is reached")
        elif (self.check_obstacle_proximity(info["laser"])):
            # If the agent hits an obstacle it gets a negative reward
            reward = -1
            terminated = True
            self.get_logger().info("The robot hits an obstacle")
        else:
            # Otherwise the episode continues
            buff = info["laser"]
            print(f"readings: {buff}")
            reward = 0
        
        return observation, reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Reset the simulation world 
        future = self.reset_world_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        # Select a random start position and orientation
        self._agent_location = np.array(random.choice(self._agent_locations), dtype=np.float32)
        # orientation = random.choice(self.orientations)
        # Set the robot to a new state with orientation, 
        # must be fixed because each position has its own orientation (not always 0° or 180°)
        # self.set_robot_state(self._agent_location, orientation)  
        self.set_robot_state(self._agent_location)
    
        # Boolean variable that waits until the set_robot_state method finishes.
        while self.robot_reset_done == False:
            rclpy.spin_once(self)

        # Sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.array(random.choice(self._agent_locations), dtype=np.float32)

        self.spin()

        # Create observation and info dictionary
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    
    def render(self): pass

    def close(self): pass
    
    def check_obstacle_proximity(self, laser_readings):
        """Check if any obstacles are too close based on laser readings."""
        for distance in laser_readings:
            if distance < self._minimum_distance_from_obstacles:
                return True
        return False

    def spin(self):
        # the node will be spun untill the sensor readings are recieived 
        self.received = False
        while (self.received == False):
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
    
    def set_robot_state(self, position):
        # Construct the request to set entity state
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = 'robot'  
        req.state.pose = Pose(
            position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
            #orientation=Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])
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
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        agent_location = np.array(self._agent_location)
        target_location = np.array(self._target_location)
        distance = np.linalg.norm(agent_location - target_location)
        return {"distance": distance, "laser": self.readings}

    def run_PPO(self):
        # checks the environment and outputs additional warnings if needed    
        check_env(self)
        self.get_logger().info("check finished")
        
        model = PPO("MultiInputPolicy", self, verbose=1)
        
        model.learn(total_timesteps=25000)
        print(f"learning is finished")
        
        # Enjoy trained agent, and test it 
        obs, info = self.reset()
        print("first reset done --------------------------------------------")
        episode_reward = 0
        true_positives = 0
        test_episodes = 20000
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
        model.save("ros2_auna/pubsub/auna_PPO")

def main(args = None):
        rclpy.init()
        auna = aunaEnvironment()
        auna.run_PPO()
        rclpy.spin(auna)
        
if __name__=="__main__": 
        main()    
