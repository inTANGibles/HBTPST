### Human Behavior Trajectory Prediction Simulation Tool in Village Transformation


**Project Overview**

This project introduces a simulation tool for addressing rural revitalization challenges, aiming to enable computers to deeply understand local environmental context and assist in decision-making during rural transformations. Our research focuses on understanding and simulating human behavioral trajectories and their perceptions of environmental elements in rural settings. By leveraging Maximum Entropy Inverse Reinforcement Learning (MEDIRL) and utilizing expert trajectories, we reconstruct reward values for various environmental features to better simulate and analyze human behavior patterns in rural areas.

# Research Methodology
1. **Preliminary Experiments**
Developed the world space configuration and state-action representation as part of an inverse reinforcement learning framework.
Represented environmental elements using a feature grid, conducting preliminary experiments on a 10x10 grid to validate the approach.
2. **Real-World Application**
Constructed human trajectories as architectural data using data from a previous project.
Classified environmental elements and integrated them into feature grids.
Trained the model using expert trajectories, enabling the agent to perceive reward values of the feature grid and simulate trajectories, providing insights into human-environment interactions.


# Installation
Ensure you have the necessary dependencies installed and run the project in a compatible environment:

```
$ conda env create -n MEIRLenv
$ conda active MEIRLenv
$ conda env create -f environment.yml
```
To run the provided Jupyter Notebooks in this project, you need to ensure Jupyter is installed and properly configured within the created Conda environment:

```
$ conda install jupyter
$ pip install jupyter
$ python -m ipykernel install --user --name=MEIRLenv
```


## Quick Start
To quickly run the project, please follow these steps after completing the installation:
1. **Preliminary Experiments**
the Location is in `demo_dmeirl/ `directory:
- `demo_DMEIRL.ipynb`: Conducts experiments with different output functions to explore initial behavior.
- `demo_eval.ipynb`: Evaluates and analyzes the results from the preliminary experiments.
2. **Real-World Application**
- `1_Train_Data_Gen.ipynb`: Expert Data Processing - Prepares and processes expert trajectories used as input data for model training.
- `2_Initializa_grid_world.ipynb`: Training Environment Setup - Sets up and configures the grid world environment necessary for training.
- `3_DMEIRL.py`: Model Training - Handles model training using inverse reinforcement learning.
- `4_DMEIRL_eval.ipynb`: Model Evaluation - Displays training results and performs evaluation.


# Contributing
-

# License
-