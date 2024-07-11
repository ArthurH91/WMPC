import os
import numpy as np
import time
import pinocchio as pin
import concurrent.futures

from wrapper_panda import PandaWrapper
from ocp import OCPPandaReachingColWithMultipleCol
from scenes import Scene
from plan_and_optimize import PlanAndOptimize

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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

num_threads = os.cpu_count()  # Get the number of CPU cores
### PARAMETERS
# Number of nodes of the trajectory
T = 10
# Time step between each node
dt = 0.01

# Name of the scene
name_scene = "box"
# Creating the robot
robot_wrapper = PandaWrapper(capsule=True)
rmodel, cmodel, vmodel = robot_wrapper()

# Creating the scene
scene = Scene()
cmodel, TARGET, q0 = scene.create_scene(rmodel, cmodel, name_scene)

n_samples = 5000

def process_sample(i):
    try:
        start = time.process_time()
        
        PaO = PlanAndOptimize(rmodel, cmodel, "panda2_leftfinger", T)
        TARGET = pin.SE3(pin.utils.rotate("x", np.pi), np.random.uniform(np.array([-0.2,-0.35 ,0.8]),np.array([0.8,0.35, 1.5]) ))
        PaO.set_ik_solver(oMgoal=TARGET)
        x0 = np.concatenate((PaO._generate_random_collision_free_configuration(), np.zeros(rmodel.nq)))
        OCP = OCPPandaReachingColWithMultipleCol(
            rmodel, cmodel, TARGET, T, dt=0.05, x0=x0
        )
        sol = PaO.solve_IK()
        PaO.init_planner(start=q0, ik_solutions=sol)
        fine_path = PaO.plan()
        t = PaO._ressample_path()
        xs, us = PaO.optimize(OCP)
        t_solve = time.process_time() - start
        xs = xs.tolist()
        Q = [x[:rmodel.nq] for x in xs]
        target_quat = pin.se3ToXYZQUATtuple(TARGET)
        return [target_quat[:3], target_quat[3:], Q], t_solve
    except:
        print(f"The generation of the trajectory {i} has failed. Trying again.")
        return None

results = []
time_calc = []

with progress_bar as p:
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_sample, i) for i in range(n_samples)]
        for future in p.track(concurrent.futures.as_completed(futures), total=n_samples):
            result = future.result()
            if result:
                res, t_solve = result
                results.append(res)
                time_calc.append(t_solve)

np.save(
    f"results/results_{name_scene}_{n_samples}.npy",
    np.array(results, dtype=object),
    allow_pickle=True,
)
np.save(
    f"results/time_result_{name_scene}_{n_samples}.npy",
    np.array(time_calc, dtype=object),
    allow_pickle=True,
)
