import sys
import os

# Add the absolute path to f1tenth_splatnav to import splat modules
splatnav_path = os.path.abspath("/f1tenth_splatnav")
sys.path.insert(0, splatnav_path)

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as Path_ros
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from skimage.graph import route_through_array
from scipy.ndimage import binary_dilation
from nav_msgs.msg import OccupancyGrid

import torch
from pathlib import Path
import numpy as np

from splat.splat_utils import GSplatLoader
from splatplan.splatplan1 import SplatPlan


class SplatPlanner2DNode(Node):
    def __init__(self):
        super().__init__('splat_planner_2d_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.goal = None

        self.resolution = None
        self.origin = None
        
        
        self.scale = 1.0 / 0.1778384719584236

        # Config path
        gsplat_config_path = Path("/f1tenth_splatnav/outputs/vicon_small/splatfacto/2025-04-22_005347/config.yml")

        # Load GSplat and Planner
        self.gsplat = GSplatLoader(gsplat_config_path, self.device)
        # self.voxel_config = {
        #     'lower_bound': torch.tensor([-1, -1, -0.35], device=self.device),
        #     'upper_bound': torch.tensor([1, 1, -0.05], device=self.device),
        #     'resolution': torch.tensor([200, 200, 3], device=self.device),
        # }
        # self.robot_config = {'radius': 0.06}
        # self.planner = SplatPlan(self.gsplat, self.robot_config, self.voxel_config, self.device)

        ## Instead of loading splat - load the 2D map generated from the splat:
        
        self.occ_map = None

        # ROS interfaces
        qos_transient = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.path_pub = self.create_publisher(Path_ros, '/splatnav_path', qos_transient)
        self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', qos_transient)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_transient)

        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        # self.clicked_sub = self.create_subscription(PointStamped, '/clicked_point', self.clicked_callback, 10)

        self.path_msg = None
        self.latest_start = None
        self.planned = False

        self.env_pcd, _, self.env_attr = self.gsplat.splat.generate_point_cloud(use_bounding_box=True)
        self.goal = self.select_goal()


    ## Function that returns a (3, ) torch tensor
    def select_goal(self):
        positives = input("Enter the object to detect (e.g., ball): ").strip()
        negatives = input("Enter negatives (if none, just press enter): ").strip()

        semantic_info = self.gsplat.splat.get_semantic_point_cloud(
            positives=positives,
            # negatives=negatives,
            pcd_attr=self.env_attr
        )

        sc_sim = torch.clip(semantic_info["similarity"] - 0.5, 0, 1)
        sc_sim = sc_sim / (sc_sim.max() + 1e-6)  # Normalize

        # Find index of maximum similarity
        max_idx = torch.argmax(sc_sim).item()

        # Get the corresponding x, y, z coordinate from the point cloud
        hottest_point = np.asarray(self.env_pcd.points)[max_idx]

        print(f"Hottest point for '{positives}' is at index {max_idx}")
        print(f"Coordinates (x, y, z): {hottest_point}: {hottest_point * self.scale}")
        goal = torch.tensor(hottest_point, dtype=torch.float32, device=self.device)

        self.publish_goal_marker(goal_pos=(goal[0].item() * self.scale, goal[1].item() * self.scale))

        return goal  # Return as (3,) torch tensor

    
    def map_callback(self, msg):
        data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.occ_map = (data == 0).astype(np.uint8)  # free = 0, occupied = 100, unknown = -1
        self.resolution = msg.info.resolution
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y) 
        self.get_logger().info("Received and processed map from /map topic.")


    def run_astar_scikit(self, start, goal, occ_map, resolution, buffer=0.3):
        def map_to_pixel(coords):
            x = coords[1]
            y = coords[0]
            x_or = self.origin[1]
            y_or = self.origin[0]
            x_pix = int((x - x_or) / self.resolution)
            y_pix = int((y - y_or) / self.resolution)
            return (x_pix, y_pix)

        def pixel_to_map(pixel):
            x_pix = pixel[0]
            y_pix = pixel[1]
            x_or = self.origin[0]
            y_or = self.origin[1]
            x = float((x_pix * self.resolution) + x_or)
            y = float((y_pix * self.resolution) + y_or)
            return (x, y)

        start_px = map_to_pixel(start)
        goal_px = map_to_pixel(goal)
        print("Start:", start, "->", start_px)
        print("Goal:", goal, "->", goal_px)

        # Inflate obstacles
        expansion_radius = int(buffer / self.resolution)
        binary_obstacles = occ_map == 0  # 0 = occupied in your logic
        structure = np.ones((2 * expansion_radius + 1, 2 * expansion_radius + 1), dtype=bool)
        inflated = binary_dilation(binary_obstacles, structure=structure)

        # Cost map
        cost_map = np.where(inflated, 1e6, 1.0).astype(np.float32)

        try:
            path, _ = route_through_array(cost_map, start_px, goal_px, fully_connected=True)
            path = np.array(path)
            path[:, [0, 1]] = path[:, [1, 0]]  # Swap back to (x, y)

            # Clip path before obstacle
            path_px = path.astype(int)
            valid_path = []
            for i, (x, y) in enumerate(path_px):
                if 0 <= y < inflated.shape[0] and 0 <= x < inflated.shape[1]:
                    if not inflated[y, x]:
                        valid_path.append(path[i])
                    else:
                        break  # stop at first point inside an obstacle
                else:
                    break  # out of bounds

            if len(valid_path) == 0:
                self.get_logger().warn("A* path found, but all points are in obstacles.")
                return {'path': None}

            path = np.array(valid_path)
            path_map = (path * self.resolution) + np.array(self.origin[:2], dtype=float)
            return {'path': path_map}

        except Exception as e:
            self.get_logger().warn(f"A* planning failed: {e}")
            return {'path': None}


    def clicked_callback(self, msg: PointStamped):
        self.goal = torch.tensor([
            msg.point.x,
            msg.point.y,
            -0.35
        ], device=self.device)

        self.get_logger().info(f"Received goal: {self.goal.cpu().numpy()}")
        self.planned = False  # allow replanning on new goal

    def odom_callback(self, msg: Odometry):
        if self.occ_map is None:
            self.get_logger().info("Waiting for occupancy map...")
            return
        
        if self.goal is None:
            self.get_logger().info("Waiting for goal from RViz...")
            return

        # Get current pose
        start = torch.tensor([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            -0.35
        ], device=self.device)

        if self.planned and self.latest_start is not None:
            # If already planned, just republish latched topics
            self.path_msg.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(self.path_msg)
            self.publish_goal_marker(goal_pos=(self.goal[0].item(), self.goal[1].item()))
            return

        self.get_logger().info(f"Planning from: {start.cpu().numpy()} to {self.goal.cpu().numpy()}")

        # output = self.planner.generate_path(start, self.goal)

        ######## 2D astar *** ########################## output - dict with key path
        print("Start: ", start)
        print("Goal: ", self.goal)
        output = self.run_astar_scikit(start.cpu().numpy(), self.goal.cpu().numpy(), self.occ_map, 1.0)

        if output['path'] is None:
            self.get_logger().warn(f"Planning from: {start.cpu().numpy()} to {self.goal.cpu().numpy()} failed")
            input("Press enter and select new waypoint on rviz to continue")
            self.goal = None
            return 

        # Convert to Path message
        self.path_msg = Path_ros()
        ## append the start pose to the path
        # Interpolate from current position to first waypoint
        # start_np = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        # first_wp = np.array([output['path'][0][0] * self.scale, output['path'][0][1] * self.scale])
        # num_interp = int(np.linalg.norm(first_wp - start_np) / 0.1)  # spacing ~10cm

        # interp_points = np.linspace(start_np, first_wp, num=num_interp + 1)

        # for pt in interp_points:
        #     pose = PoseStamped()
        #     pose.header.frame_id = 'map'
        #     pose.pose.position.x = pt[0]
        #     pose.pose.position.y = pt[1]
        #     pose.pose.position.z = 0.0
        #     pose.pose.orientation.w = 1.0
        #     self.path_msg.poses.append(pose)

        
        self.path_msg.header.frame_id = 'map'
        self.path_msg.header.stamp = self.get_clock().now().to_msg()

        for pt in output['path']:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            self.path_msg.poses.append(pose)

        self.path_pub.publish(self.path_msg)
        self.publish_goal_marker(goal_pos=(self.goal[0].item(), self.goal[1].item()))
        self.get_logger().info("Published new path and goal marker.")

        self.latest_start = start
        self.planned = True

    def publish_goal_marker(self, goal_pos):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = goal_pos[0]
        marker.pose.position.y = goal_pos[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime.sec = 0
        self.goal_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = SplatPlanner2DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
