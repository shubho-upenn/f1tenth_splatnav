import open3d
import numpy as np


# ply_file_path = "data/vicon/sparse_pc.ply"
# ply_file_path = "data/splatnav_test/sparse_pc.ply"
ply_file_path = "outputs/vicon_small/mesh.ply"
mesh = open3d.io.read_point_cloud(ply_file_path)

axes = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

aabb = mesh.get_axis_aligned_bounding_box()
aabb.color = (0,1,0)
print(aabb)
# open3d.visualization.draw_geometries([mesh, axes, aabb])

# Define min/max bounds (np arrays of shape (3,))
'''
vicon - sparse_pc
min_xyz = np.array([-10.0861, -14.4966, -6.0])
max_xyz = np.array([11.3945, 14.1342, 7.5])
'''
'''
min_xyz = np.array([-10.0861, -14.4966, -6.0])
max_xyz = np.array([11.3945, 14.1342, 7.5])
'''

min_xyz = np.array([-1, -1, -0.35])
max_xyz = np.array([1, 1, 0])
# Clip points within bounds
points = np.asarray(mesh.points)
colors = np.asarray(mesh.colors) if mesh.has_colors() else None

mask = np.all((points >= min_xyz) & (points <= max_xyz), axis=1)
clipped_points = points[mask]

# Create filtered point cloud
clipped_pcd = open3d.geometry.PointCloud()
clipped_pcd.points = open3d.utility.Vector3dVector(clipped_points)

if colors is not None:
    clipped_pcd.colors = open3d.utility.Vector3dVector(colors[mask])

# Bounding box (visualization only)
bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=min_xyz, max_bound=max_xyz)
bbox.color = (0, 1, 0)
print(bbox)

# Axes at origin
axes = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

# Visualize everything
open3d.visualization.draw_geometries([clipped_pcd, axes, bbox])
