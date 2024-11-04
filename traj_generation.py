from typing import Tuple
import os
import torch
import pinocchio as pin
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.visualizer import create_viewer, add_sphere_to_viewer
from utils.wrapper_panda import PandaWrapper
from utils.ocp import OCP
from utils.param_parsers import ParamParser
from utils.plan_and_optimize import PlanAndOptimize

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


class TrajGeneration:
    def __init__(self, rmodel: pin.Model, cmodel: pin.GeometryModel, pp: ParamParser):
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.pp = pp
        self.PaO = PlanAndOptimize(self.rmodel, self.cmodel, "panda2_hand_tcp", pp.T)

    def generate_traj(self, X0, targ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a single trajectory and returns inputs and outputs as tensors."""
        try:
            ocp_creator = OCP(
                self.rmodel, self.cmodel, TARGET_POSE=targ, x0=X0, pp=self.pp
            )
            ocp = ocp_creator.create_OCP()
            xs, _ = self.PaO.compute_traj(X0[:7], targ, ocp)
            target = pin.SE3ToXYZQUAT(targ)[:3]
            return target, xs
        except Exception as e:
            print(f"Failed to generate trajectory. Error: {str(e)}")
            return None, None

    @staticmethod
    def from_targ_xs_to_input_output(
        targ: np.ndarray, xs: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts target and state trajectory xs into input/output tensors."""
        X0 = xs[0]
        inputs_tensor = torch.tensor(
            np.concatenate((targ, X0[:7])), dtype=torch.float32
        )
        outputs = np.array([X[:7] for X in xs[1:]])
        outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
        return inputs_tensor, torch.flatten(outputs_tensor)

    def save_results_as_tensors(self, results, filename="trajectories.pt"):
        """Stores generated results as a tensor file using torch.save."""
        torch.save(results, filename)
        print(f"Trajectories stored as {filename}.")


if __name__ == "__main__":

    ### NUMBER OF TRAJS FOR GENERATION
    num_trajs = 10

    ### FILENAME TO SAVE TRAJS
    filename = "results/trajectories.pt"
    # Initialize robot model and parameters
    robot_wrapper = PandaWrapper(capsule=False)
    rmodel, cmodel, vmodel = robot_wrapper()

    yaml_path = os.path.join(os.path.dirname(__file__), "scenes.yaml")
    pp = ParamParser(yaml_path, 1)
    # initial_config = pp.initial_config
    initial_config = None
    cmodel = pp.add_collisions(rmodel, cmodel)
    vis = create_viewer(rmodel, cmodel, cmodel)
    targ = pp.target_pose
    TG = TrajGeneration(rmodel, cmodel, pp)

    results = []
    with progress_bar as p:
        for i in p.track(range(num_trajs)):
            # targ = TG.PaO.get_random_reachable_target()
            X0 = np.concatenate(
                (
                    (
                        pin.randomConfiguration(rmodel)
                        if initial_config is None
                        else initial_config
                    ),
                    np.zeros(rmodel.nv),
                )
            )
            target, xs = TG.generate_traj(X0, targ)
            if xs is not None:
                input_, output = TG.from_targ_xs_to_input_output(target, xs)
                results.append((input_, output))

    torch.save(results, filename)

    for i, result in enumerate(results):
        if i > 0:
            vis.viewer[f"goal{i-1}"].delete()
        add_sphere_to_viewer(vis, f"goal{i}", 5e-2, result[0][:3].numpy(), color=0x006400)
        vis.display(result[0][3:].numpy())
        input()
        for x in torch.split(result[1], rmodel.nq):
            vis.display(x[:7].numpy())
            input()
