import os.path as osp
from torch.utils.data import DataLoader
import torch

from mlp_training import MLP, TrajectoriesDataset

# from model import Net
from utils.wrapper_panda import PandaWrapper
from utils.param_parsers import ParamParser
from utils.visualizer import create_viewer, add_sphere_to_viewer



#### Load data ####
results_dir = osp.join(osp.dirname(str(osp.abspath(__file__))), "results")
data_filename = "trajectories_sc2_rs_n1000.pt"
data_path = osp.join(results_dir, "trajectories" ,  data_filename)
data = torch.load(data_path, weights_only=True)
    
T = 15
nq = 7

# Create dataset
dataset = TrajectoriesDataset(data)
# Paths to the model and data
model_file_name = "trajectories_sc2_rs_n1000_model.pth"
model_path = osp.join(results_dir, "models", model_file_name)
net = MLP(hidden_sizes=[128, 128, 64])


# Load the model state
if osp.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
else:
    print("Model file does not exist.")
    exit()

# Create dataset and dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Set the model to evaluation mode
net.eval()


robot_wrapper = PandaWrapper(capsule=True)
rmodel, cmodel, vmodel = robot_wrapper()

yaml_path = "scenes.yaml"
pp = ParamParser(yaml_path, 2)
cmodel = pp.add_collisions(rmodel, cmodel)

vis = create_viewer(rmodel, cmodel, cmodel)
add_sphere_to_viewer(
    vis, "goal", 5e-2, pp.target_pose.translation, color=0x006400
)
### INITIAL X0


q0 = data[1][0][3:] + 0.3*torch.randn(7)
inputs = torch.cat((data[1][0][:3], q0))
with torch.no_grad():
    output = net(inputs)

print("visualisation of the input given trajectory")

while True:
    print("visualisation of the NN given trajectory")
    vis.display(q0.numpy())
    input()
    for xs in output.split(7):
        vis.display(xs.numpy())
        input() 
    print("replay")
