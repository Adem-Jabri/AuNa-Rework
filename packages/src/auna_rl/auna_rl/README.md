## Setup and Running the Agent

### 1. Load the Model
To utilize the best-performing agent, please load the `ppo_model_18000_steps.zip` file from this folder. Update the `model_path` variable in the `main` method of the trainer file to match the location of this file.
### 2. Build the Package
To build the `auna_rl` package, execute the following command in your terminal:
colcon build --packages-select auna_rl
### 3. **Configure the environment**:
   To set up the environment for running the agent, execute the following command:
   source packages/install/setup.bash

### 4. **Run the agent**:
To start the agent, use the following command: ros2 run auna_rl trainer

## Additional Information

For more details about the objective, methodology, challenges, solutions and the results of this project, refer to the autonomous_racing_using_reinforcement_learning.pdf file in this directory.

