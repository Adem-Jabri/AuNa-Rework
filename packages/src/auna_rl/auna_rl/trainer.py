import os
import rclpy
from rclpy.node import Node
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from auna_rl.auna_env import aunaEnvironment
import threading

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
        self.log_dir = "./logs18/"
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

        if model_path:
            self.model = PPO.load(model_path, env=self.env)
            self.model.learning_rate = 0.0001
            self.model.gamma=0.6
        else:
            # PPO model configuration
            self.model = PPO("MultiInputPolicy", self.env, verbose=1, tensorboard_log=self.log_dir,
                            batch_size=256, n_steps=4096, n_epochs=10,
                            learning_rate=0.0001, ent_coef=0.01, clip_range=0.1,
                            gamma=0.6, gae_lambda=0.95,
                            policy_kwargs={'net_arch': [400, 300]})

        # Callbacks
        self.checkpoint_callback = CheckpointCallback(save_freq=3000, save_path='./models18/', name_prefix='ppo_model')
        self.eval_callback = EvalCallback(self.auna, best_model_save_path='./models18/', log_path=self.log_dir, eval_freq=3000,
                                          deterministic=True, render=False)
        self.progress_callback = ProgressCallback(check_freq=20000)

    def start_training(self):
        self.training_thread = threading.Thread(target=self.train)
        self.training_thread.start()

    def train(self):
        # Start the training process with callbacks for saving the model, evaluation, and progress tracking
        self.model.learn(total_timesteps=20000, callback=[self.checkpoint_callback, self.eval_callback, self.progress_callback])
        self.get_logger().info("Training completed")
        
        # Save the model and close environment
        self.model.save(os.path.expanduser("~/AuNa-Rework/packages/src/auna_rl"))
        self.get_logger().info("Model saved successfully")
    
    def test_model(self, num_episodes=10):
        for episode in range(num_episodes):
            obs, info = self.auna.reset()
            episode_rewards = 0
            done = False
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.auna.step(action)
                episode_rewards += reward
                if done or truncated:
                    break
            self.get_logger().info(f"Episode {episode + 1}: Total Reward: {episode_rewards}")

def main(args=None):
    rclpy.init(args=args)
    model_path = "/home/vscode/workspace/models17/ppo_model_18000_steps.zip"
    trainer = PPOTrainingNode(model_path)
    
    #trainer.start_training()
    trainer.test_model(num_episodes=10)
    rclpy.spin(trainer)  # Ensure that rclpy.spin() references the correct node object

    trainer.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
