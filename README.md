gym_rocketlander/PPO.py: This file trains and runs the PPO agent. Set watch to False to watch a pretrained model, and True to train a new model
gym_rocketlander/__init__.py: Registers the gym environment
gym_rocketlander/envs/rocket_lander.py: This file creates the gym environemnt. Contains functions to initialize, reset, render, and step through the environment
base_model.zip: Stores the full trained base model
increating_heights.zip: Stores the fully trained model that was trained on increasing heights
upward_penalty.zip: Stores the fully trained agent with the upward penalty
drift.zip: Stores the fully trained model with a denser reward function
final_model.zip: Stores the fully trained final model
