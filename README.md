# WMPC

## Overview
Warmstart for MPC using memory of motion.

## Architecture of the repo

### Folders
- */models* where the URDF / SRDF description of the robot are.
- */utils* for the helper functions.
- */results* where the models and the trajectories generated are.

### Useful files
- *traj_generation.py* is the script used to generate trajectories. To generate trajectories, use the parser: For instance, `python traj_generation.py -sc 2 -rs -rt -n 1000 -d` will generate 1000 trajectories (-n) in the scene 2 (-sc) (defined after), with a random reachable target (-rt) and a random start (-rs), and at the end, will display them in meshcat (-d). The trajectories generated are saved with the same naming.
- *mlp_training.py*, self explainatory. You can change the number of layers and the data in the script.
- *eval.py*, generate a starting configuration (random or not) and evaluate the model to generate the trajectory.
- *data_visualisation.ipynb* is the notebook to visualize the trajectories generated by the planner.
- *scenes.yaml* is used to define the scene.

## Installation
To use this code, you need: 

- [colmpc](https://github.com/agimus-project/colmpc/tree/main) and its dependencies
-  

## Usage
To start the project, use the following command:
```bash
npm start
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
