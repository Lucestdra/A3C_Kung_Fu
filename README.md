# A3C_Kung_Fu
This repository contains the implementation of a Deep Reinforcement Learning (DRL) agent that plays the Atari game KungFuMaster using the Actor-Critic method with a Convolutional Neural Network (CNN). The project is built with PyTorch and uses the Gymnasium framework for environment simulation.
# Features
- Convolutional Neural Network (CNN): The agent's neural network is designed with convolutional layers to process the visual input from the Atari game environment.
- Actor-Critic Architecture: The model leverages an Actor-Critic approach to balance learning of both the policy (actions) and value (state) functions.
- Frame Preprocessing: The environment observations are preprocessed to grayscale and resized to a smaller resolution, and a stack of four frames is used to capture the dynamics of the environment.
- Multi-Environment Training: Parallel training is conducted across multiple environments, allowing the agent to learn more efficiently from diverse experiences.
- Video Recording: The repository includes functionality to record and display the agent's gameplay, providing visual feedback on the agent's performance.

# Installation

To run the code, you need to have Python installed along with the following dependencies. 
You can install them by following these steps:

- Install Gymnasium:
pip install gymnasium

- Install the Atari environment and accept the ROM license:
pip install "gymnasium[atari, accept-rom-license]"

- Install SWIG (needed for Box2D environments):
apt-get install -y swig

- Install the Box2D environment:
pip install gymnasium[box2d]
