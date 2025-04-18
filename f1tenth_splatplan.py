#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
import json
# from SFC.corridor_utils import SafeFlightCorridor
from splat.splat_utils import GSplatLoader
from splatplan.splatplan1 import SplatPlan
# from splatplan.spline_utils import SplinePlanner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)

path_to_GSplat = Path("outputs/splatnav_test/gemsplat/2025-04-17_024254/config.yml")       
# path_to_GSplat = Path("/splats/outputs/splatnav_test/splatfacto/2025-04-08_183001/config.yml")
lower_bound = torch.tensor([-1.33, -0.5, -0.17], device=device)     ## from 'flight' example =  mac_schw lab 
upper_bound = torch.tensor([1, 0.5, 0.26], device=device)           ## from 'flight' example =  mac_schw lab
resolution = torch.tensor([100, 100, 100], device=device)           ## in mac_schw lab it is int(100), but can be a torch.tensor vector (3,)
voxel_config = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'resolution': resolution
               }

robot_radius = 0.02
robot_config = {'radius': robot_radius}

tnow = time.time()
gsplat = GSplatLoader(path_to_GSplat, device)
print('Time to load GSplat:', time.time() - tnow)

tnow = time.time()
mesh = gsplat.save_mesh("outputs/splatnav_test//mesh.ply")
print('Time to save GSplat:', time.time() - tnow)

'''
planner = SplatPlan(gsplat, robot_config, voxel_config, device)

x =             ## start (current pose of car)
goal =          ## goal

output = planner.generate_path(x, goal)     ## add some code in this or before this to ensure that the path generated can be traversed by the vehicle
'''
