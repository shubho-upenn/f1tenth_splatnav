import open3d as o3d
import numpy as np

# --- Configuration ---
test_name = "splatnav_test/"
test_dir = "outputs/" + test_name

# Load point cloud
pcd = o3d.io.read_point_cloud(test_dir + "mesh.ply")

# Load path (N, 3)
path = np.load(test_dir + "path_astar_2d.npy")

# Bounding box from PyTorch tensors
bbox_min = lower_bound = np.array([-1.33, -0.5, -0.17])     ## from 'flight' example =  mac_schw lab 
bbox_max = np.array([1, 0.5, 0.26])  # Example min

# Filter point cloud within the bounding box
pcd_points = np.asarray(pcd.points)
pcd_colors = np.asarray(pcd.colors)  # <-- also get colors

mask = np.all((pcd_points >= bbox_min) & (pcd_points <= bbox_max), axis=1)

pcd.points = o3d.utility.Vector3dVector(pcd_points[mask])
pcd.colors = o3d.utility.Vector3dVector(pcd_colors[mask])  # <-- restore filtered colors


# Create path LineSet
points = path.tolist()
lines = [[i, i+1] for i in range(len(points) - 1)]
colors = [[1, 0, 0] for _ in lines]  # Red

path_line_set = o3d.geometry.LineSet()
path_line_set.points = o3d.utility.Vector3dVector(points)
path_line_set.lines = o3d.utility.Vector2iVector(lines)
path_line_set.colors = o3d.utility.Vector3dVector(colors)

# Spheres along path
spheres = []
for i, point in enumerate(path):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.translate(point)
    if i == 0:
        sphere.paint_uniform_color([0, 1, 0])  # Start - Green
    elif i == len(path) - 1:
        sphere.paint_uniform_color([0, 0, 1])  # Goal - Blue
    else:
        sphere.paint_uniform_color([1, 0, 0])  # Intermediate - Red
    spheres.append(sphere)

# Show visualization
o3d.visualization.draw_geometries([pcd, path_line_set] + spheres)
