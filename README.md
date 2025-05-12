# F1-tenth Implementation of SplatNav (Semantic navigation)

Run in the following docker:

"
docker pull ghcr.io/shubho-upenn/splatnav_gemsplat_ros:latest
"

To create Semantic Gaussian Splat Map: refer to https://github.com/chengine/splatnav to create GEmSplat map using nerfstudio

To generate 3D dense pointcloud/collision boundaries: use https://github.com/shubho-upenn/f1tenth_splatnav/blob/main/f1tenth_splatplan.py with mapping = True

To generate 2D map from 3D dense pointcloud: use https://github.com/shubho-upenn/f1tenth_splatnav/blob/main/convert_pcl_to_2d_map.py

ROS package with various scripts is in ros2_ws. Need to clone f1_tenth_gym_bridge in workspace and put in the mp files generated from above into appropriate folder.

The various planner nodes generate a Path message, that Pure Pursuit can follow
