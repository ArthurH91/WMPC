# Import all packages
import numpy as np
import pydynorrt as pyrrt
from pydynorrt import pin_more as pyrrt_vis # Small utility functions to visualize motion planning
import pinocchio as pin  # Not needed in the current script (consider removing if not used)
import meshcat
import time
# Pydynorrt supports planning in any combination of R^n, SO(3), and SO(2).
# Let's solve a problem in R^3 x SO(3) (=SE(3)) with an optimal planner.

lb = np.array([-2, -2, -1, -1, -1, -1, -1])
ub = np.array([3, 2, 1, 1, 1, 1, 1])
# State is R^3 and quaternion, with the real part last
start = np.array([0, 0, 0, 0, 0, 0, 1])
goal = np.array([2, 1, 1, 0, 0, 0, 1])
urdf = pyrrt.DATADIR + "models/se3_window.urdf"
srdf = pyrrt.DATADIR + "models/se3_window.srdf"

cm = pyrrt.Collision_manager_pinocchio()
cm.set_urdf_filename(urdf)
cm.set_srdf_filename(srdf)
cm.build()
assert cm.is_collision_free(start)
assert cm.is_collision_free(goal)

# Let's use RRT star

rrt = pyrrt.PlannerRRTStar_Combined()
rrt.init(7)
rrt.set_state_space_with_string(["Rn:3", "SO3"])

rrt.set_start(start)
rrt.set_goal(goal)


config_str = """
[RRTStar_options]
max_it = 20000
goal_bias = 0.1
collision_resolution = 0.05
max_step = 1.0
max_compute_time_ms = 3e3 # One second of compute time
goal_tolerance = 0.001
max_num_configs = 20000
"""

rrt.read_cfg_string(config_str)
rrt.set_is_collision_free_fun_from_manager(cm)
rrt.set_bounds_to_state(lb, ub)
tic = time.time()
out = rrt.plan()
toc = time.time()
path = rrt.get_path()
fine_path = rrt.get_fine_path(0.05)
planner_data = rrt.get_planner_data()
# Add a small sleep to give time for the std::cout inside the compiled library to appear on the screen
# before we print from Python.
time.sleep(0.001)
print(
    "Planning Time [s] (note that we are running an asymptotically optimal planner)", toc - tic)
# We can examine the content of planner data
print("Fields in planner_data", [i for i in planner_data])

# we can display all the paths found by the planner,
# using different values of transparency.
paths = [[np.array(x) for x in path] for path in planner_data["paths"]]

viewer = meshcat.Visualizer()
viewer_helper = pyrrt_vis.ViewerHelperRRT(viewer, urdf, "", start, goal)
robot = viewer_helper.robot
viz = viewer_helper.viz

idx_vis_name = "base_link"
IDX_VIS = viewer_helper.robot.model.getFrameId(idx_vis_name)

display_count = 0  # just to enumerate the number of edges
for pp, _path in enumerate(paths[:-1]):
    transparency = (pp+1) / (len(paths))
    for i in range(len(_path) - 1):
        q1 = _path[i]
        q2 = _path[i + 1]
        pyrrt_vis.display_edge(robot, q1, q2, IDX_VIS, display_count, viz, radius=0.02,
                                    color=[0.1, 0.1, 0.1, 0.8 * transparency])
        display_count += 1

# Best path in Blue
for i in range(len(path) - 1):
    q1 = path[i]
    q2 = path[i + 1]
    pyrrt_vis.display_edge(robot, q1, q2, IDX_VIS, display_count, viz, radius=0.02,
                          color=[0.0, 0.0, 1.0, 0.5])
    display_count += 1

input()
# Finally, we can visualize the path of the robot :)
for p in fine_path:
    viz.display(np.array(p))
    time.sleep(0.01)
    input()