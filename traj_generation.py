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


if __name__ == "__main__":
    import argparse

    ### PARSER ###
    parser = argparse.ArgumentParser(description="Trajectory generation parser")

    parser.add_argument(
        "-n",
        "--num_trajs",
        type=int,
        default=10,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "-rs",
        "--random_initial_start",
        action="store_true",
        help="Flag to use a random initial start",
    )
    parser.add_argument(
        "-rt",
        "--random_target",
        action="store_true",
        help="Flag to use a random target",
    )
    parser.add_argument(
        "-d",
        "--display-traj",
        action="store_true",
        help="Flag to display trajectories in a meshcat viewer",
    )
    parser.add_argument(
        "-sc", "--scene", type=int, default=1, help="Number of the scene to use"
    )
    args = parser.parse_args()

    ### INITIALIZE ROBOT ###
    robot_wrapper = PandaWrapper(capsule=True)
    rmodel, cmodel, vmodel = robot_wrapper()
    yaml_path = os.path.join(os.path.dirname(__file__), "scenes.yaml")
    pp = ParamParser(yaml_path, args.scene)
    cmodel = pp.add_collisions(rmodel, cmodel)
    TG = TrajGeneration(rmodel, cmodel, pp)

    ### STARTING POINT & ENDING POINT OF THE TRAJ ###(THEY WILL BE MODIFIED IF RANDOM ARGS IN PARSER)
    initial_config = pp.initial_config
    targ = pp.target_pose

    ### TRAJECTORY GENERATION ###
    results = []
    with progress_bar as p:
        for i in p.track(range(args.num_trajs), description="Generating trajectories"):
            if args.random_initial_start:
                initial_config = pin.randomConfiguration(rmodel)
            if args.random_target:
                targ = TG.PaO.get_random_reachable_target()
            X0 = np.concatenate(
                (
                    (initial_config),
                    np.zeros(rmodel.nv),
                )
            )
            target, xs = TG.generate_traj(X0, targ)
            if xs is not None:  # ie if the generation was successful
                input_, output = TG.from_targ_xs_to_input_output(target, xs)
                results.append((input_, output))

    ### SAVE TRAJS ###
    filename = (
        f"results/trajectories/trajectories_sc{args.scene}"
        + ("_rs" if args.random_initial_start else "")
        + ("_rt" if args.random_target else "")
        + f"_n{args.num_trajs}.pt"
    )
    torch.save(results, filename)

    ### DISPLAY TRAJS ###
    if args.display_traj:
        vis = create_viewer(rmodel, cmodel, cmodel)
        for i, result in enumerate(results):
            if i > 0:
                vis.viewer[f"goal{i-1}"].delete()
            add_sphere_to_viewer(
                vis, f"goal{i}", 5e-2, result[0][:3].numpy(), color=0x006400
            )
            vis.display(result[0][3:].numpy())
            input()
            for x in torch.split(result[1], rmodel.nq):
                vis.display(x[:7].numpy())
                input()
