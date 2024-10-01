import yaml
import numpy as np

import pinocchio as pin
import hppfcl 


class ParamParser:
    def __init__(self, path: str, scene: int):
        self.path = path
        self.params = None
        self.scene = scene
        
        with open(self.path, "r") as stream:
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
        else:
            raise ValueError(f"Unknown shape {shape}")
        
    
    def add_collisions(self, rmodel: pin.Model, cmodel: pin.GeometryModel):
        """ Add collisions to the robot model

        Args:
            rmodel (pin.Model): Robot model
            cmodel (pin.GeometryModel): Collision model

        Returns:
            cmodel: Collision model with added collisions
        """

        for obs in self.data["OBSTACLES"]:
            obs_hppfcl = self._parse_obstacle_shape(self.data["OBSTACLES"][obs]["type"], self.data["OBSTACLES"][obs]["dim"])
            Mobs = pin.SE3(pin.Quaternion(*tuple(self.data["OBSTACLES"][obs]["orientation"])).toRotationMatrix(), np.array(self.data["OBSTACLES"][obs]["translation"]))
            obs_id_frame = rmodel.addFrame(pin.Frame(obs, 0, 0, Mobs, pin.OP_FRAME))
            obs_geom = pin.GeometryObject(obs, 0, 0, rmodel.frames[obs_id_frame].placement, obs_hppfcl)
            obs_geom.meshColor = np.concatenate((np.random.randint(0,1, 3), np.ones(1))) 
            _ = cmodel.addGeometryObject(obs_geom)
                                                  
        for col in self.data["collision_pairs"]:
            if cmodel.existGeometryName(col[0]) and cmodel.existGeometryName(col[1]):
                cmodel.addCollisionPair( pin.CollisionPair(
                    cmodel.getGeometryId(col[0]),
                    cmodel.getGeometryId(col[1]),
                )
                )
            else:
                raise ValueError(f"Collision pair {col} does not exist in the collision model")
        return cmodel
    
    def get_target_pose(self):
        return pin.SE3(pin.Quaternion(*tuple(self.data["TARGET_POSE"]["orientation"])).toRotationMatrix(), np.array(self.data["TARGET_POSE"]["translation"]))
    
    def get_initial_config(self):
        return np.array(self.data["INITIAL_CONFIG"])
    
    def get_X0(self):
        return np.concatenate((self.get_initial_config(),np.array(self.data["INITIAL_VELOCITY"])))
    
    def get_safety_threshold(self):
        return self.data["SAFETY_THRESHOLD"]
    
    def get_T(self):
        return self.data["T"]
    
    def get_dt(self):
        return self.data["dt"]
    
    def get_di(self):
        return self.data["di"]
    
    def get_ds(self):
        return self.data["ds"]
    
    def get_ksi(self):
        return self.data["ksi"]
    
    def get_W_xREG(self):
        return self.data["WEIGHT_xREG"]
    
    def get_W_uREG(self):
        return self.data["WEIGHT_uREG"]
    
    def get_W_gripper_pose(self):
        return self.data["WEIGHT_GRIPPER_POSE"]
    
    def get_W_gripper_pose_term(self):
        return self.data["WEIGHT_GRIPPER_POSE_TERM"]
    


if __name__ == "__main__":
    from wrapper_panda import PandaWrapper
    
    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=True)
    rmodel, cmodel, vmodel = robot_wrapper()
    
    path = "scenes.yaml"
    scene = 0
    pp = ParamParser(path, scene)
    cmodel = pp.add_collisions(rmodel, cmodel)
