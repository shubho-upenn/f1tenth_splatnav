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
# import tf_transformations
from scipy.spatial.transform import Rotation as R

from splat.splat_utils import GSplatLoader
# from splatplan.splatplan1 import SplatPlan


import heapq
import math

class HybridNode:
    def __init__(self, x, y, yaw, g_cost=0.0, h_cost=0.0, parent=None):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):  # Needed for heapq
        return self.f_cost < other.f_cost




class SplatPlanner2DNode(Node):
    def __init__(self):
        super().__init__('splat_planner_2d_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.goal = None

        self.resolution = None
        self.origin = None
        
        
        self.scale = (1.0 / 0.1778384719584236) / 1.78

        # Config path
        gsplat_config_path = Path("/f1tenth_splatnav/outputs/vicon_working/gemsplat/2025-04-19_192539/config.yml")

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
        # odom = '/ego_racecar/odom'
        odom = '/pf/pose/odom'

        self.odom_sub = self.create_subscription(Odometry, odom, self.odom_callback, 10)
        # self.clicked_sub = self.create_subscription(PointStamped, '/clicked_point', self.clicked_callback, 10)

        self.path_msg = None
        self.latest_start = None
        self.planned = False

        self.env_pcd, _, self.env_attr = self.gsplat.splat.generate_point_cloud(use_bounding_box=True)
        self.goal = self.select_goal()

    def map_callback(self, msg):
        data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        binary_map = (data == 0).astype(np.uint8)  # free = 0, occupied = 100, unknown = -1
        free_ratio = np.mean(binary_map)
        print(free_ratio)
        car_radius = 0.15  # meters (adjust to your car’s clearance)
        inflation_radius = int(car_radius / msg.info.resolution)

        structure = np.ones((2 * inflation_radius + 1, 2 * inflation_radius + 1), dtype=np.uint8)
        inflated = binary_dilation(binary_map == 0, structure=structure)

        # Free cells = 1, Obstacle cells = 0
        self.occ_map = (~inflated).astype(np.uint8)
        free_ratio = np.mean(self.occ_map)
        print(free_ratio)
        self.resolution = msg.info.resolution
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y) 
        self.get_logger().info("Received and processed map from /map topic.")


    def run_hybrid_astar(self, start, goal, occ_map, resolution, origin, wheelbase=0.3):
        steering_angles = np.linspace(-0.52, 0.52, 25)
        directions = [0.0707]  # forward only
        goal_tolerance = 0.1

        def map_to_world(px, py):
            return (px * resolution + origin[0], py * resolution + origin[1])

        def world_to_map(x, y):
            return int((x - origin[0]) / resolution), int((y - origin[1]) / resolution)

        def is_goal(n):
            dx = n.x - goal[0]
            dy = n.y - goal[1]
            dyaw = n.yaw - goal[2]
            # return math.sqrt(dx**2 + dy**2 + dyaw**2) < goal_tolerance
            return math.sqrt(dx**2 + dy**2) < goal_tolerance


        def heuristic(n):
            dx = n.x - goal[0]
            dy = n.y - goal[1]
            return math.sqrt(dx**2 + dy**2)

        def is_valid(x, y):
            px, py = world_to_map(x, y)
            if 0 <= px < occ_map.shape[1] and 0 <= py < occ_map.shape[0]:
                return occ_map[py, px] == 1  # free
            return False

        def expand_node(n):
            next_nodes = []
            for d in directions:
                for sa in steering_angles:
                    dx = d * math.cos(n.yaw)
                    dy = d * math.sin(n.yaw)
                    dyaw = d * math.tan(sa) / wheelbase

                    nx = n.x + dx
                    ny = n.y + dy
                    nyaw = n.yaw + dyaw

                    if not is_valid(nx, ny):
                        continue

                    g_new = n.g_cost + math.hypot(dx, dy)
                    h_new = heuristic(n)
                    new_node = HybridNode(nx, ny, nyaw, g_new, h_new, parent=n)
                    next_nodes.append(new_node)
            return next_nodes

        open_set = []
        start_node = HybridNode(*start, g_cost=0.0, h_cost=heuristic(HybridNode(*start)))
        heapq.heappush(open_set, start_node)
        visited = set()

        while open_set:
            current = heapq.heappop(open_set)
            if is_goal(current):
                path = []
                while current:
                    path.append([current.x, current.y])
                    current = current.parent
                # print(path[::-1])
                return {'path': np.array(path[::-1]).astype(float)}
            
            key = (int(current.x / resolution), int(current.y / resolution), int(current.yaw * 10))
            if key in visited:
                continue
            visited.add(key)

            for n in expand_node(current):
                heapq.heappush(open_set, n)

        return {'path': None}


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



    def clicked_callback(self, msg: PointStamped):
        self.goal = torch.tensor([
            msg.point.x,
            msg.point.y,
            -0.35
        ], device=self.device)

        # self.goal_aligned = 

        self.get_logger().info(f"Received goal: {self.goal.cpu().numpy()}")
        self.planned = False  # allow replanning on new goal

    def odom_callback(self, msg: Odometry):
        if self.occ_map is None:
            self.get_logger().info("Waiting for occupancy map...")
            return
        
        if self.goal is None:
            # self.get_logger().info("Waiting for goal from RViz...")
            return

        # Get current pose
        start = torch.tensor([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            -0.35
        ], device=self.device)

        # def quat_to_yaw(q):
        #     return tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        pose = msg.pose.pose
        orientation_q = pose.orientation
        r = R.from_quat([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        yaw = r.as_euler('xyz')[2]  # Extract yaw (rotation about Z)
        start_aligned = torch.tensor([pose.position.x, pose.position.y, yaw], device=self.device)

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
        # output = self.run_hybrid_astar(start.cpu().numpy(), self.goal.cpu().numpy(), self.occ_map, 1.0)
        output = self.run_hybrid_astar(
                                        start=start_aligned.cpu().numpy(),
                                        goal=self.goal.cpu().numpy(),
                                        occ_map=self.occ_map,
                                        resolution=self.resolution,
                                        origin=self.origin
                                    )

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
            print(pt)
            print(type(pt[0]))
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
