import open3d as o3d
import numpy as np

# --- Configuration ---
test_name = "vicon_small/"
test_dir = "outputs/" + test_name

# Load point cloud
pcd_raw = o3d.io.read_point_cloud(test_dir + "mesh.ply")

# Visualize raw point cloud
print("Showing original point cloud...")
o3d.visualization.draw_geometries([pcd_raw])

# --- Outlier Removal ---
# Option 1: Statistical Outlier Removal
pcd, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# Option 2: Radius Outlier Removal (alternative)
# pcd, ind = pcd_raw.remove_radius_outlier(nb_points=16, radius=0.05)

# --- Bounding Box Crop ---
bbox_min = np.array([-1, -1, -0.4])
bbox_max = np.array([1, 1, -0.05])

pcd_points = np.asarray(pcd.points)
pcd_colors = np.asarray(pcd.colors)
mask = np.all((pcd_points >= bbox_min) & (pcd_points <= bbox_max), axis=1)

pcd.points = o3d.utility.Vector3dVector(pcd_points[mask])
pcd.colors = o3d.utility.Vector3dVector(pcd_colors[mask])

# --- Load and Draw Path ---
path = np.load(test_dir + "path_astar_2d.npy")
points = path.tolist()
lines = [[i, i + 1] for i in range(len(points) - 1)]
colors = [[1, 0, 0] for _ in lines]

path_line_set = o3d.geometry.LineSet()
path_line_set.points = o3d.utility.Vector3dVector(points)
path_line_set.lines = o3d.utility.Vector2iVector(lines)
path_line_set.colors = o3d.utility.Vector3dVector(colors)

# --- Spheres Along the Path ---
spheres = []
for i, point in enumerate(path):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.translate(point)
    if i == 0:
        sphere.paint_uniform_color([0, 1, 0])  # Start
    elif i == len(path) - 1:
        sphere.paint_uniform_color([0, 0, 1])  # Goal
    else:
        sphere.paint_uniform_color([1, 0, 0])  # Intermediate
    spheres.append(sphere)

# --- Final Visualization ---
print("Showing cleaned + cropped point cloud...")
o3d.visualization.draw_geometries([pcd, path_line_set] + spheres)
