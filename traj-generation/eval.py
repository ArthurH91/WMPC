import os
import numpy as np
import torch
import pinocchio as pin

from model import Net



class Eval:
    "Evaluation class that takes the two configurations in input and returns a trajectory linking both of the configurations."

    def __init__(self, model_path: str, data_path: str) -> None:
        """Instantiate the class that takes the two configurations in input and returns a trajectory linking both of the configurations.

        Args:
            model_path (str): path of the trained model.
            data_path (str): path of the generated data.

        Raises:
            NameError: Wrong model path.
        """
        data = np.load(
            data_path,
            allow_pickle=True,
        )

        self._T = len(data[0, 2])
        self._nq = len(data[0, 2][0])
        self._net = Net(self._nq, self._T)

        # Load the model state
        if os.path.exists(model_path):
            self._net.load_state_dict(torch.load(model_path))
            print("Model loaded successfully.")
        else:
            raise NameError("Model file does not exist.")

        # Set the model to evaluation mode
        self._net.eval()

    def generate_trajectory(self, q0, target):
        target_quat = pin.SE3ToXYZQUATtuple(TARGET)
        target_array = np.concatenate((target_quat[:3], target_quat[3:]))
        inputs = np.concatenate((q0, target_array))
        with torch.no_grad():
            output = self._net(torch.tensor(inputs, dtype=torch.float32))
        output = output.numpy()[0]
        X = [
            np.concatenate((output[k], np.zeros(self._nq))) for k in range(len(output))
        ]
        for i, qv in enumerate(X):
            # Initial speed is equal to 0.
            if not i == 0: 
                qv[-self._nq :] = (
                    qv[self._nq :] - X[i - 1][self._nq :]
                )  # The speed is simply the difference between each configuration.

        return X


if __name__ == "__main__":

    from wrapper_panda import PandaWrapper
    from visualizer import create_viewer
    from scenes import Scene

    model_path = "/home/arthur/Desktop/Code/WMPC/traj-generation/nn_models/box_5000.pth"
    data_path = (
        "/home/arthur/Desktop/Code/WMPC/traj-generation/results/results_box_5000.npy"
    )

    eval = Eval(model_path, data_path)

    ### PARAMETERS
    # Number of nodes of the trajectory
    T = 10
    # Time step between each node
    dt = 0.01

    # Name of the scene
    name_scene = "box"

    # Creating the robot
    robot_wrapper = PandaWrapper(auto_col=True, capsule=True)
    rmodel, cmodel, vmodel = robot_wrapper()

    # Creating the scene
    scene = Scene()
    cmodel, TARGET, q0 = scene.create_scene(rmodel, cmodel, name_scene)

    # Generating the meshcat visualizer
    MeshcatVis = create_viewer(rmodel, cmodel , vmodel)

    MeshcatVis.display(q0)
    MeshcatVis.display()

    output = eval.generate_trajectory(q0, TARGET)

    while True:
        print("visualisation of the NN given trajectory")
        for xs in output:
            MeshcatVis.display(xs[:rmodel.nq])
            input()
        print("replay")
