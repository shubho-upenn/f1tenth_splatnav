import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_occupancy_grid_from_ply(ply_path, z_min, z_max, grid_resolution=0.05,
                                   bbox_min=None, bbox_max=None):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)  # (N, 3)

    # Apply global scale if needed
    scale = 1.0 / 0.1778384719584236

    # Apply bounding box filter
    if bbox_min is not None and bbox_max is not None:
        in_bbox = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
        points = points[in_bbox]

    # Slice Z range
    slice_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    slice_points = slice_points * scale  # Scale after slicing

    if slice_points.shape[0] == 0:
        raise ValueError("No points found in the given Z slice.")

    # Compute grid size and origin
    min_x, min_y = np.min(slice_points[:, :2], axis=0)
    max_x, max_y = np.max(slice_points[:, :2], axis=0)

    grid_width = int(np.ceil((max_x - min_x) / grid_resolution)) + 1
    grid_height = int(np.ceil((max_y - min_y) / grid_resolution)) + 1

    # Initialize grid
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Convert world coords → grid indices
    x_idx = ((slice_points[:, 0] - min_x) / grid_resolution).astype(int)
    y_idx = ((slice_points[:, 1] - min_y) / grid_resolution).astype(int)

    occupancy_grid[y_idx, x_idx] = 1

    # Origin in world coordinates (meters)
    origin = (min_x, min_y)
    return occupancy_grid, origin

# def save_pgm(grid, filename):
#     flipped = np.flipud(grid) * 255  # Convert to grayscale and flip for ROS
#     img = Image.fromarray(flipped.astype(np.uint8), mode='L')
#     img.save(filename)

def save_pgm(grid, filename):
    flipped = np.flipud(1 - grid) * 255  # Invert occupancy: 1 → 0, 0 → 1
    padded = np.pad(flipped, pad_width=1, mode='constant', constant_values=1)
    img = Image.fromarray(padded.astype(np.uint8), mode='L')
    img.save(filename)

def save_yaml(filename, image_file, resolution, origin):
    with open(filename, 'w') as f:
        f.write(f"image: {image_file}\n")
        f.write(f"resolution: {resolution}\n")
        f.write(f"origin: [{origin[0]:.6f}, {origin[1]:.6f}, 0.0]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.5\n")
        f.write("free_thresh: 0.2\n")

if __name__ == "__main__":
    ply_file = "/home/shubho/splatnav_f1tenth/outputs/vicon_small/mesh.ply"
    bbox_min = np.array([-1.0, -1.0, -0.4])
    bbox_max = np.array([ 1.0,  1.0, -0.05])
    z_min = -0.27
    z_max = -0.23
    resolution = 0.05

    pgm_filename = "/home/shubho/ese6150/sim_ws/src/f1tenth_gym_ros/maps/vicon_small.pgm"
    yaml_filename = "/home/shubho/ese6150/sim_ws/src/f1tenth_gym_ros/maps/vicon_small.yaml"

    # Generate occupancy grid and origin
    grid, origin = create_occupancy_grid_from_ply(
        ply_file, z_min, z_max, resolution,
        bbox_min=bbox_min, bbox_max=bbox_max
    )

    # Visualize
    plt.imshow(np.flipud(grid), cmap='gray')
    plt.title("2D Occupancy Grid")
    plt.axis("off")
    plt.show()

    # Save outputs
    save_pgm(grid, pgm_filename)
    save_yaml(yaml_filename, os.path.basename(pgm_filename), resolution, origin)

    print(f"Saved map to {pgm_filename} and {yaml_filename}")
