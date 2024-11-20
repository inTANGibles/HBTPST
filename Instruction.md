# Instruction File for Reproducing Results in "Your Manuscript Title"



## 1. Introduction
This document provides step-by-step instructions to reproduce the findings reported in our manuscript, including figures, tables, and measures presented in Sections 3 and 4. 
1. **Simulation Experiments** 
Developed the world space configuration and state-action representation as part of an inverse reinforcement learning framework.
Represented environmental elements using a feature grid, conducting preliminary experiments on a 10x10 grid to validate the approach.
2. **Real-World Application**
Constructed human trajectories as architectural data using data from a previous project.
Classified environmental elements and integrated them into feature grids.
Trained the model using expert trajectories, enabling the agent to perceive reward values of the feature grid and simulate trajectories, providing insights into human-environment interactions.

Please follow the steps below to ensure successful reproduction of the results.


## 2. Prerequisites and Setup
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


## 3. Data Preparation

**Simulation Experiments** ：
The simulated data is generated for testing and validating the model's performance in controlled scenarios.
**Real-World Application** ：
The real-world data was collected and is stored in the `wifi_track_data/dacang` directory.
- `grid_data`: Contains all initial data on environmental states, including manual labeling of environmental elements (referenced in Fig. 12). This data represents and describes the distribution and state of various elements in either simulated or actual environments.
- `imgs`: Contains actual maps of the formal site, used for visualization or analysis related to the real-world site.
- `origin_data`: Represents the initial measured data collected during the data acquisition phase, without any processing applied.
- `pos_data`: Based on the initial measured data, this dataset includes added relative geographic coordinates, allowing for spatial positioning and further computational analysis.
- `temp_data`: Temporary data generated during the data processing workflow, used for storing intermediate results or cache data, facilitating further processing and optimization.
- `track_data`: Contains the final data used for model training, after filtering, processing, and preparation for training purposes.


## 4. Steps to Reproduce Sections 3 "Simulation Experiments"
- Figures 5 (b)(c), 6 (a)(b), and 8 (a) , Table 1 can be reproduced using demo_demirl.ipynb.
- Figures 9 (A)(b) can be reproduced using demo_eval.ipynb.
See the comments in the corresponding document for detailed steps.

## 5. Steps to Reproduce Sections 4 "Real-World Application"
- Figures 10 (a)(b) can be reproduced using 1_Train_Data_Gen.ipynb.
- Figures 13 (a)(b), 14(b) can be reproduced using 2_initialized_grid.ipynb.
- Figures 15 (a) can be reproduced using 4_DMEIRL_eval.ipynb.
- The analysis of the results and visualization of the trajectories are placed in `data_mining`
See the comments in the corresponding document for detailed steps.