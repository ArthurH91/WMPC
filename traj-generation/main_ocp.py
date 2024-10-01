import numpy as np
import pinocchio as pin

from wrapper_panda import PandaWrapper
from visualizer import create_viewer, add_sphere_to_viewer, add_cube_to_viewer
from param_parsers import ParamParser

import create_ocp

# Creating the robot
robot_wrapper = PandaWrapper(capsule=True)
rmodel, cmodel, vmodel = robot_wrapper()

pp = ParamParser("scenes.yaml", 0)

cmodel = pp.add_collisions(rmodel, cmodel)

cdata = cmodel.createData()
rdata = rmodel.createData()

vis = create_viewer(
    rmodel, cmodel, cmodel
)

# Generating the meshcat visualizer
vis = create_viewer(
    rmodel, cmodel, cmodel
)
add_sphere_to_viewer(vis, "goal", 5e-2,  pp.get_target_pose().translation, color=0x006400)

# OCP with distance constraints
OCP_dist = create_ocp.create_ocp_distance(rmodel, cmodel, pp)
OCP_dist.solve()
print("OCP with distance constraints solved")

# OCP with velocity constraints
ocp_vel = create_ocp.create_ocp_velocity(rmodel, cmodel, pp)
ocp_vel.solve()
print("OCP with velocity constraints solved")
for i, xs in enumerate(ocp_vel.xs):
        q = np.array(xs[:7].tolist())
        pin.framesForwardKinematics(rmodel, rdata, q)
        add_cube_to_viewer(
            vis,
            "vcolmpc" + str(i),
            [2e-2, 2e-2, 2e-2],
            rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation,
            color=100000000,
        )

for i, xs in enumerate(OCP_dist.xs):
        q = np.array(xs[:7].tolist())
        pin.framesForwardKinematics(rmodel, rdata, q)
        add_sphere_to_viewer(
            vis,
            "colmpc" + str(i),
            2e-2,
            rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation,
            color=100000,
        )
while True:
    print("OCP with distance constraints")
    for q in OCP_dist.xs:
        vis.display(q[:7])
        input()
    print("OCP with velocity constraints")
    for q in ocp_vel.xs:
        vis.display(q[:7])
        input()