import os
import rclpy
from rclpy.node import Node
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecVideoRecorder
from stable_baselines3.common.utils import get_schedule_fn
from auna_rl.auna_env import aunaEnvironment
import threading
import logging

# Custom callback for progress tracking
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            print(f"Step: {self.num_timesteps}, Mean Reward: {np.mean(self.model.rollout_buffer.rewards)}")
        return True

class PPOTrainingNode(Node):
    def __init__(self, model_path=None):
        super().__init__('ppo_training_node')
        self.get_logger().info("Setting up the environment")

        # Set NumPy to raise exceptions for floating-point errors
        np.seterr(all='raise')
        # Enable PyTorch anomaly detection for debugging
        th.autograd.set_detect_anomaly(True)

        self.auna = aunaEnvironment()
        check_env(self.auna)

        # Environment setup with NaN detection
        env = lambda: aunaEnvironment()
        self.env = DummyVecEnv([env])
        self.env = VecCheckNan(self.env, raise_exception=True)

        # Configure logger
        self.log_dir = "./logs7/"
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

        if model_path:
            self.model = PPO.load(model_path, env=self.env)
            # Dynamically adjust learning rate and other adjustable parameters
            self.model.learning_rate = get_schedule_fn(0.0001)  # Adjusted learning rate
            self.model.gamma = 0.90
            #self.clip_range = 0.1
            self.ent_coef = 0.015
        else:
            # PPO model configuration
            self.model = PPO("MultiInputPolicy", self.env, verbose=1, tensorboard_log=self.log_dir,
                            batch_size=64, n_steps=2048, n_epochs=10,
                            learning_rate=0.001, ent_coef=0.01, clip_range=0.1,
                            gamma=0.99, gae_lambda=0.95,
                            policy_kwargs={'net_arch': [400, 300]})

        if hasattr(self.env, 'render_mode'):
            self.env.render_mode = 'rgb_array'
        else:
            raise NotImplementedError("Environment must support rgb_array rendering for video recording.")
    
        # Video recording setup
        self.env = VecVideoRecorder(self.env, "./videos/",
                                    record_video_trigger=lambda step: step % 15000 == 0,
                                    video_length=1200*50,  # 20 minutes of recording at 50 FPS
                                    name_prefix="ppo-training")

        # Callbacks
        self.checkpoint_callback = CheckpointCallback(save_freq=3000, save_path='./models7/', name_prefix='ppo_model')
        self.eval_callback = EvalCallback(self.auna, best_model_save_path='./models7/', log_path=self.log_dir, eval_freq=3000,
                                          deterministic=True, render=False)
        self.progress_callback = ProgressCallback(check_freq=20000)

    def start_training(self):
        self.training_thread = threading.Thread(target=self.train)
        self.training_thread.start()

    def train(self):
        # Start the training process with callbacks for saving the model, evaluation, and progress tracking
        self.model.learn(total_timesteps=400000, callback=[self.checkpoint_callback, self.eval_callback, self.progress_callback])
        self.get_logger().info("Training completed")

        # Test the trained agent
        #obs, info = self.auna.reset()
        #episode_rewards = []
        #true_positives = 0
        #test_episodes = 5000
        #for _ in range(test_episodes):
        #    action, _states = self.model.predict(obs, deterministic=True)
        #    obs, reward, done, truncated, info = self.auna.step(action)
        #    episode_rewards.append(reward)
        #    if info["done"]:
        #        true_positives += 1
        #        obs, info = self.auna.reset()

        #accuracy = true_positives / test_episodes
        #self.get_logger().info(f"Test Accuracy: {accuracy*100:.2f}%")

        # Save the model and close environment
        self.model.save(os.path.expanduser("~/AuNa-Rework/packages/src/auna_rl"))
        self.get_logger().info("Model saved successfully")

def main(args=None):
    rclpy.init(args=args)
    model_path = "/home/vscode/workspace/models3/ppo_model_174000_steps.zip"
    trainer = PPOTrainingNode()
    trainer.start_training()
    rclpy.spin(trainer)  # Ensure that rclpy.spin() references the correct node object

    #if trainer.training_thread.is_alive():
    #    trainer.training_thread.join()

    trainer.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
