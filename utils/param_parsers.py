# BSD 3-Clause License
#
# Copyright (C) 2024, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

import hppfcl
import numpy as np
import pinocchio as pin
import yaml


class ParamParser:
    def __init__(self, path: str, scene: int):
        self.path = path
        self.params = None
        self.scene = scene

        with open(self.path) as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.data = self.params["scene" + str(self.scene)]

    @staticmethod
    def _parse_obstacle_shape(shape: str, size: list):
        if shape == "box":
            return hppfcl.Box(*size)
        elif shape == "sphere":
            return hppfcl.Sphere(size[0])
        elif shape == "cylinder":
            return hppfcl.Cylinder(size[0], size[1])
        elif shape == "ellipsoid":
            return hppfcl.Ellipsoid(*size)
        elif shape == "capsule":
            return hppfcl.Capsule(*size)
        else:
            raise ValueError(f"Unknown shape {shape}")

    def _add_ellipsoid_on_robot(self, rmodel: pin.Model, cmodel: pin.GeometryModel):
        """Add ellipsoid on the robot model"""
        if "ROBOT_ELLIPSOIDS" in self.data:
            for ellipsoid in self.data["ROBOT_ELLIPSOIDS"]:
                rob_hppfcl = hppfcl.Ellipsoid(
                    *self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["dim"]
                )
                idf_rob = rmodel.getFrameId(
                    self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["parentFrame"]
                )
                print(idf_rob)
                idj_rob = rmodel.frames[idf_rob].parentJoint
                if (
                    "translation" in self.data["ROBOT_ELLIPSOIDS"][ellipsoid]
                    and "orientation" in self.data["ROBOT_ELLIPSOIDS"][ellipsoid]
                ):
                    rot_mat = (
                        pin.Quaternion(
                            *tuple(
                                self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["orientation"]
                            )
                        )
                        .normalized()
                        .toRotationMatrix()
                    )
                    Mrob = pin.SE3(
                        rot_mat,
                        np.array(
                            self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["translation"]
                        ),
                    )
                else:
                    Mrob = rmodel.frames[idf_rob].placement
                rob_geom = pin.GeometryObject(
                    ellipsoid, idj_rob, idf_rob, Mrob, rob_hppfcl
                )
                rob_geom.meshColor = np.r_[1, 1, 0, 1]
                cmodel.addGeometryObject(rob_geom)
        return cmodel

    def add_collisions(self, rmodel: pin.Model, cmodel: pin.GeometryModel):
        """Add collisions to the robot model"""
        cmodel = self._add_ellipsoid_on_robot(rmodel, cmodel)
        for obs in self.data["OBSTACLES"]:
            obs_hppfcl = self._parse_obstacle_shape(
                self.data["OBSTACLES"][obs]["type"], self.data["OBSTACLES"][obs]["dim"]
            )
            Mobs = pin.SE3(
                pin.Quaternion(*tuple(self.data["OBSTACLES"][obs]["orientation"]))
                .normalized()
                .toRotationMatrix(),
                np.array(self.data["OBSTACLES"][obs]["translation"]),
            )
            obs_id_frame = rmodel.addFrame(pin.Frame(obs, 0, 0, Mobs, pin.OP_FRAME))
            obs_geom = pin.GeometryObject(
                obs, 0, obs_id_frame, rmodel.frames[obs_id_frame].placement, obs_hppfcl
            )
            obs_geom.meshColor = np.concatenate(
                (np.random.randint(0, 1, 3), np.ones(1))
            )
            _ = cmodel.addGeometryObject(obs_geom)

        for col in self.data["collision_pairs"]:
            if cmodel.existGeometryName(col[0]) and cmodel.existGeometryName(col[1]):
                cmodel.addCollisionPair(
                    pin.CollisionPair(
                        cmodel.getGeometryId(col[0]),
                        cmodel.getGeometryId(col[1]),
                    )
                )
            else:
                raise ValueError(
                    f"Collision pair {col} does not exist in the collision model"
                )
        return cmodel

    @property
    def target_pose(self):
        return pin.SE3(
            pin.Quaternion(
                *tuple(self.data["TARGET_POSE"]["orientation"])
            ).toRotationMatrix(),
            np.array(self.data["TARGET_POSE"]["translation"]),
        )

    @property
    def initial_config(self):
        return np.array(self.data["INITIAL_CONFIG"])

    @property
    def X0(self):
        return np.concatenate(
            (self.initial_config, np.array(self.data["INITIAL_VELOCITY"]))
        )

    @property
    def safety_threshold(self):
        return self.data["SAFETY_THRESHOLD"]

    @property
    def T(self):
        return self.data["T"]

    @property
    def dt(self):
        return self.data["dt"]

    @property
    def di(self):
        return self.data["di"]

    @property
    def ds(self):
        return self.data["ds"]

    @property
    def ksi(self):
        return self.data["ksi"]

    @property
    def W_xREG(self):
        return self.data["WEIGHT_xREG"]

    @property
    def W_uREG(self):
        return self.data["WEIGHT_uREG"]

    @property
    def W_gripper_pose(self):
        return self.data["WEIGHT_GRIPPER_POSE"]

    @property
    def W_gripper_pose_term(self):
        return self.data["WEIGHT_GRIPPER_POSE_TERM"]


if __name__ == "__main__":
    import os.path as osp
    import example_robot_data as robex
    from visualizer import create_viewer, add_sphere_to_viewer
    # Creating the robot
    # Creating the robot
    panda = robex.load("panda_collision")
    rmodel, cmodel, vmodel = panda.model, panda.collision_model, panda.visual_model

    yaml_path = osp.join(osp.dirname(osp.dirname(__file__)), "scenes.yaml")
    pp = ParamParser(yaml_path, 1)

    geom_models = [vmodel, cmodel]
    rmodel, geometric_models_reduced = pin.buildReducedModel(
        rmodel,
        list_of_geom_models=geom_models,
        list_of_joints_to_lock=[7,8],
        reference_configuration=np.append(np.array(pp.initial_config), np.zeros(2)))
    # geometric_models_reduced is a list, ordered as the passed variable "geom_models" so:
    vmodel, cmodel = geometric_models_reduced[
        0], geometric_models_reduced[1]
    cmodel = pp.add_collisions(rmodel, cmodel)

    cdata = cmodel.createData()
    rdata = rmodel.createData()
    vis = create_viewer(rmodel, cmodel, cmodel)
    add_sphere_to_viewer(
        vis, "goal", 5e-2, pp.target_pose.translation, color=0x006400
    )
    vis.display(pp.initial_config)