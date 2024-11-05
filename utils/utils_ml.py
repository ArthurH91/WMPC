import numpy as np
import matplotlib.pyplot as plt 

import pinocchio as pin

import matplotlib.colors as colors
import matplotlib.cm as cm

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def set_progress_bar():
    """ Set the progress bar for the rich library.

    Returns:
        Progress : Progress bar object.
    """
    # Define custom progress bar
    progress_bar = Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )    
    
    return progress_bar

def convert_q_traj_tensor_to_xs(q_traj_tensor: tuple) -> list:
    """ Convert a trajectory of joint angles to a trajectory of states usable by crocoddyl.

    Args:
        q_traj_tensor (tuple): Tuple of torch tensors representing the joint angles of the trajectory.

    Returns:
        list: List of numpy arrays representing the states of the trajectory.
    """
    XS = []
    for q in q_traj_tensor:
        XS.append(np.concatenate((q.detach().numpy(), np.zeros(7))))
    return XS

def convert_inputs_outputs_to_trajs(inputs: list, outputs: list) -> list:
    """ Convert inputs and outputs stored in .pt file as [target, q0] and [q1, .. , qT-1] respectively to a list of trajectories.

    Args:
        inputs (torch.Tensor): Tensor of inputs.
        outputs (torch.Tensor): Tensor of outputs.

    Returns:
        list: List of numpy arrays representing the trajectories.
    """
    trajs = [inputs[3:].detach().numpy()]
    for q in (outputs.split(7)):
        trajs.append(q.detach().numpy())
    return trajs


def plot_trajs(rmodel: pin.Model, trajs_list: list, costs_list: list = None, title: str = "Trajectories"):
    """ Plot a list of trajectories.

    Args:
        rmodel (pin.Model): Pinocchio model of the robot.
        trajs_list (list): List of numpy arrays representing the trajectories.
        costs_list (list): List of costs representing the values of cost for each trajectories.
        title (str, optional): Title of the plot. Defaults to "Trajectories".
    """    
    
    rdata = rmodel.createData()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap
    if costs_list is not None:
        cmap = plt.get_cmap('inferno')
        norm = plt.Normalize(np.min(costs_list), np.max(costs_list))  # Normalize the cost values
    
    for i, traj in enumerate(trajs_list):
        trajs = []
        for iter, q in enumerate(traj):
            pin.forwardKinematics(rmodel, rdata, q[:rmodel.nq])
            pin.updateFramePlacements(rmodel, rdata)
            pos = rdata.oMf[rmodel.getFrameId("panda2_hand_tcp")].translation
            
            # Add labels for the first trajectory and the start position
            if i == 0 and iter == 0:
                legend_start = 'Start of trajectory'
            elif i == 0 and iter == 1:
                legend_traj = 'Nodes of trajectory'
            else:
                legend_start = ''
                
            if iter == 0:
                ax.scatter(pos[0], pos[1], pos[2], c='g', marker='o', label=legend_start)
            trajs.append([pos[0], pos[1], pos[2]])
        
        # Get the color based on the cost
        color = cmap(norm(costs_list[i])) if costs_list is not None else 'b'  # Default to blue if no costs provided
        
        # Plot the trajectory
        ax.plot([p[0] for p in trajs], [p[1] for p in trajs], [p[2] for p in trajs], label=legend_traj if i == 0 else '', color=color)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End effector position for each trajectory')
    if costs_list is not None:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Value of cost function')
    ax.legend()
    plt.show()



def compare_trajs(rmodel: pin.Model, trajs_ocp_list: list, trajs_ml_list:list):
    """ Compare the trajectories from the optimal control problem and the machine learning model.
    
    Args:
        rmodel (pin.Model): Pinocchio model of the robot.
        trajs_ocp_list (list): List of numpy arrays representing the trajectories from the optimal control problem.
        trajs_ml_list (list): List of numpy arrays representing the trajectories from the machine learning model.
    """
    
    rdata = rmodel.createData()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, traj in enumerate(trajs_ml_list):
        trajs = []
        for iter, q in enumerate(traj):
            pin.forwardKinematics(rmodel, rdata, q[:rmodel.nq])
            pin.updateFramePlacements(rmodel, rdata)
            pos = rdata.oMf[rmodel.getFrameId("panda2_hand_tcp")].translation
            
            # Add labels for the first trajectory and the start position
            if i == 0 and iter == 0:
                legend_start = 'Start of trajectory'
            elif i == 0 and iter == 1:
                legend_traj = 'Nodes from ML'
            else:
                legend_start = ''
                
            if iter == 0:
                ax.scatter(pos[0], pos[1], pos[2], c='g', marker='o', label=legend_start)
            trajs.append([pos[0], pos[1], pos[2]])
                
        # Plot the trajectory
        ax.plot([p[0] for p in trajs], [p[1] for p in trajs], [p[2] for p in trajs], label=legend_traj if i == 0 else '', color="g")
    
    for i, traj in enumerate(trajs_ocp_list):
        trajs = []
        for iter, q in enumerate(traj):
            pin.forwardKinematics(rmodel, rdata, q[:rmodel.nq])
            pin.updateFramePlacements(rmodel, rdata)
            pos = rdata.oMf[rmodel.getFrameId("panda2_hand_tcp")].translation
            
            # Add labels for the first trajectory and the start position
            if i == 0 and iter == 1:
                legend_traj = 'Nodes from OCP'
            else:
                legend_start = ''
                
            if iter == 0:
                ax.scatter(pos[0], pos[1], pos[2], c='g', marker='o', label=legend_start)
            trajs.append([pos[0], pos[1], pos[2]])
                
        # Plot the trajectory
        ax.plot([p[0] for p in trajs], [p[1] for p in trajs], [p[2] for p in trajs], label=legend_traj if i == 0 else '', color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End effector position for each trajectory')
    ax.legend()
    plt.show()