Master's Research in Computational Neuroscience
This repository contains the code and materials for my master's research in Computational Neuroscience at Wits University. The research focuses on creating a system that processes an image, identifies reward points, and generates a reward map in a 2D discrete space. The project uses reinforcement learning and autoencoders to analyze and model reward structures in a given environment.

Table of Contents
Overview
Installation
Usage
Project Structure
Research Details
Dependencies
License
Overview
The goal of this research is to create a system that:

Uses an autoencoder to map a reward space.
Analyzes the reward points of an environment (e.g., MiniGrid).
Outputs a 2D reward map.
Uses a trained agent to build the Successor Representation (SR) of the space and then build value functions from it.
Core Concepts
Autoencoder: A type of neural network used for dimensionality reduction and feature extraction.
Successor Representation (SR): A method used in reinforcement learning to model the expected future states of an agent’s environment.
MiniGrid: A simple grid-based environment used for reinforcement learning tasks.
Installation
To run this project, you need to set up the environment and install dependencies. Follow these steps:

Clone the repository:

git clone https://github.com/yourusername/masters-research.git
cd masters-research
Create a virtual environment (optional, but recommended):

python -m venv venv
Activate the virtual environment:

On Windows:
.\venv\Scripts\activate
On Mac/Linux:

source venv/bin/activate
Install the dependencies:

pip install -r requirements.txt
Usage
Training the agent: To start training the agent in the environment, run:

python main.py
Generating reward maps: After training the agent, the reward maps are automatically generated. You can visualize the results using Matplotlib or save them using plt.imsave().

Loading pre-trained models: If you have pre-trained models or success representations, you can load them using the appropriate functions in the code.

Project Structure
masters-research/
│
├── agents/                    # Folder containing agent scripts (e.g., `RandomAgent`, `SuccessorAgent`)
├── models/                    # Folder containing model scripts (e.g., autoencoders, SR computation)
├── results/                   # Folder for saving output (e.g., images, models, reward maps)
├── datasets/                  # Folder for datasets
├── main.py                    # Main script to run the project
├── requirements.txt           # List of required Python packages
├── README.md                  # This file
└── utils.py                   # Helper functions
Research Details
This research aims to address how reinforcement learning models can efficiently represent reward structures and environments. The following methods are employed:

Data Collection: Sample images are collected from the environment for training.
Autoencoder Training: The autoencoder is trained on these images to learn a compressed representation.
Successor Representation (SR): The SR is built by running an agent through the environment, enabling it to predict future states and rewards.
Dependencies
This project relies on the following Python packages:

numpy
matplotlib
gym (for the environment)
tensorflow (for neural networks)
scikit-learn (for various utilities)
torch (for PyTorch models, if applicable)
You can install all required dependencies by running:

pip install -r requirements.txt
License
This project is licensed under the MIT License - see the LICENSE file for details.
