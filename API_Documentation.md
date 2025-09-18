# GridWorld Toolkit API Documentation

## Core Classes

### GridWorld
**File**: `grid_world.py`

#### Initialization
```python
GridWorld(
    environments_img_folderPath=None,    # Environment image folder path
    environments_arr=None,              # Environment array
    features_folderPath=None,           # Features folder path
    states_features=None,               # State features dictionary
    expert_traj_filePath=None,          # Expert trajectory file path
    expert_trajs=None,                  # Expert trajectory dataframe
    width=100, height=75,               # Grid dimensions
    trans_prob=0.6,                     # Transition probability
    discount=0.98,                      # Discount factor
    active_all=False,                   # Activate all states
    manual_deact_states=[],             # Manually deactivated states
    real_reward_mat=[],                 # Real reward matrix
    traj_length_bias=0                  # Trajectory length bias
)
```

#### Key Methods
| Method | Description | Returns |
|--------|-------------|---------|
| `GetCountGrid()` | Get grid visit counts | `np.ndarray` |
| `GetAllActiveStates()` | Get active states | `list` |
| `GetTransitionMat()` | Get transition matrix | `np.ndarray` |
| `CoordToState(coord)` | Convert coordinates to state | `int` |
| `StateToCoord(state)` | Convert state to coordinates | `tuple` |
| `ShowEnvironments()` | Display environment grids | `None` |
| `ShowFeatures()` | Display feature grids | `None` |
| `ShowRewardsResult(rewards, title)` | Display reward results | `None` |

### GridWorld_trajGen
**File**: `trajGen_grid_world.py`

#### Initialization
```python
GridWorld_trajGen(
    width, height,                      # Grid dimensions
    states_matrix=None,                 # State reward matrix
    init_states=[],                     # Initial states list
    features_folderPath=None,           # Features folder path
    rewards_mul=[],                     # Reward multipliers
    n_objects=-1, n_colors=-1,          # Number of objects and colors
    trans_prob=0.9,                     # Transition probability
    discount=0.9,                       # Discount factor
    model=None                          # Neural network model
)
```

#### Key Methods
| Method | Description | Returns |
|--------|-------------|---------|
| `OptimalPolicy(rewards_arr)` | Compute optimal policy | `np.ndarray` |
| `GenerateTrajectories(traj_count, traj_length, policy, save)` | Generate expert trajectories | `pd.DataFrame` |
| `reset(random=True)` | Reset environment | `int` |
| `step(a)` | Execute action | `int` |

### Experts
**File**: `experts.py`

#### Initialization
```python
Experts(
    width, height,                      # Grid dimensions
    trajs_file_path=None,               # Trajectory file path
    df_trajs=None,                      # Trajectory dataframe
    bias=0                              # Length bias
)
```

#### Key Methods
| Method | Description | Returns |
|--------|-------------|---------|
| `GetExpertTraj(m)` | Get specific trajectory | `list` |
| `GetExpertsMovingCenter()` | Get expert movement centers | `tuple` |
| `ReadCluster(c_result)` | Read clustering results | `None` |
| `ApplyCluster(c_set)` | Apply clustering | `None` |

### DataParser
**File**: `data_parser.py`

#### Initialization
```python
DataParser(
    df_wifipos=None,                    # WiFi position data
    df_path=None,                       # Path data
    width=100, height=75                # Grid dimensions
)
```

#### Key Methods
| Method | Description | Returns |
|--------|-------------|---------|
| `PathToStateActionPairs(df, scale=1)` | Convert paths to state-action pairs | `DataFrame` |
| `ParseEnvironmentFromFolder(folder_path)` | Parse environment from folder | `None` |
| `ParseEnvironmentFromImage(image, feature_name, save_path)` | Parse environment from image | `None` |

## Utility Functions

### grid_utils
| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `DrawPathOnGrid(grid, point1, point2)` | Draw path on grid | `grid, point1, point2` | `np.ndarray` |
| `StatesToStateActionPairs(states)` | Convert states to state-action pairs | `states` | `list` |
| `CoordToState(coord, width)` | Convert coordinates to state | `coord, width` | `int` |
| `StateToCoord(state, width)` | Convert state to coordinates | `state, width` | `tuple` |

### grid_plot
| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `ShowGridWorld(grid, width, height, title)` | Display grid world | `grid, width, height, title` | `None` |
| `ShowGridWorld_anime(grids, width, height, title)` | Display grid world animation | `grids, width, height, title` | `None` |
| `ShowTraj(track, width, height, title)` | Display trajectory | `track, width, height, title` | `None` |

## Data Formats

### Trajectory Data Format
```python
# CSV format
{
    'm': [1, 2, 3, ...],                    # Trajectory ID
    'trajs': [                              # Trajectory data
        [(state, action, next_state), ...], # State-action-next_state triplets
        ...
    ]
}
```

### Environment Data Format
```python
# Environment dictionary format
{
    'feature_name': np.ndarray,             # Feature name: 2D array
    ...
}

# State features format
{
    state_id: [feature1, feature2, ...],    # State ID: feature values list
    ...
}
```

## Usage Examples

### Basic Usage
```python
# Create grid world
grid_world = GridWorld(
    environments_img_folderPath="env_images/",
    expert_traj_filePath="trajectories.csv"
)

# Display environment
grid_world.ShowEnvironments()
grid_world.ShowFeatures()

# Generate trajectories
traj_gen = GridWorld_trajGen(
    width=100, height=75,
    features_folderPath="features/",
    rewards_mul=[1.0, -0.5, 0.8]
)

trajectories = traj_gen.GenerateTrajectories(
    traj_count=100,
    traj_length=50
)
```
