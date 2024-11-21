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

- Steps to Reproduce Figures 5(b)(c), 6(a)(b), 8(a), and Table 1
    1. Open `demo_demirl.ipynb`.
    2. Run the first three cells to Configuring the environment and importing data.
    3. Run the code in **Cell [3]** to generate `Figures 5 (b)`.
    4. Run the code in **Cell [4]** to generate `Table 1`.
    5. Run the code in **Cell [5]** to generate `Figures 6 (a)(b)`.
    6. Run the code in **Cell [6] [7] [8]** to generate `Figures 8 (a)`.

- Steps to Reproduce Figures 9 (a)(b).
    1. Open `demo_eval.ipynb`.
    2. Run the first three cells to Configuring the environment and importing data.
    3. Run the code in **Cell [4]** to generate `Figures 9 (a)`.
    4. Run the code in **Cell [5]** to generate `Figures 9 (b)`.


## 5. Steps to Reproduce Sections 4 "Real-World Application"

- Steps to Reproduce Figures 10 (a)(b) 
    1. Open `1_Train_Data_Gen.ipynb`.
    2. Run the first three cells to Configuring the environment and importing data.Ensure the input data `dacang_track_data_final.csv` exists in the `df` directory.Ensure the input data `wifi_pos.csv` exists in the `df_wifipos` directory.Ensure the input data `path_pos.csv` exists in the `df_path` directory.
    3. Run the code in **Cell [5] [6]** to generate `Figures 10 (a)`, which takes a particular mac(mac_list[283]) value for example.
    4. Run the code in **Cell [11]** to generate `Figures 10 (b)`.

- Steps to Reproduce Figures 13 (a)(b), 14(b) 
    1. Open `2_initialized_grid.ipynb`.
    2. Run the first three cells to Configuring the environment and importing data.Ensure the input data `env_imgs/40_30` exists in the `env_folder_path` directory.Ensure the input data `trajs_0117_40x30.csv` exists in the `expert_traj_path` directory.
    3. Run the code in **Cell [5] [6]** to generate `Figures 14(b)`.
    4. Run the code in **Cell [11] [14]** to generate `Figures 13 (a)(b)`.

- Steps to Reproduce Figures 15 (a) 
    1. Open `4_DMEIRL_eval.ipynb`.
    2. Run the first two cells to Configuring the environment and using tanh to train the model.
    3. Run all codes and the conclusion of DMEIRL to generate `Figures 15 (a)`


- The analysis of the results and visualization of the trajectories are placed in `data_mining`
See the comments in the corresponding document for detailed steps.