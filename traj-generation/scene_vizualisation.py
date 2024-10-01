import yaml
import numpy as np
import pinocchio as pin

from wrapper_panda import PandaWrapper
from visualizer import create_viewer
from param_parsers import ParamParser

# Creating the robot
robot_wrapper = PandaWrapper(capsule=True)
rmodel, cmodel, vmodel = robot_wrapper()

pp = ParamParser("scenes.yaml", 0)

cmodel = pp.add_collisions(rmodel, cmodel)

vis = create_viewer(
    rmodel, cmodel, cmodel
)

vis.display(pin.randomConfiguration(rmodel))