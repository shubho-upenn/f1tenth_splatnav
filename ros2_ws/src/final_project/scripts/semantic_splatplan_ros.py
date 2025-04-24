import sys
import os
# from header import NeRF

# Add the absolute path to f1tenth_splatnav to import splat modules
splatnav_path = os.path.abspath("/f1tenth_splatnav")
sys.path.insert(0, splatnav_path)

import gc
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as Path_ros
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import torch
from pathlib import Path
import numpy as np

from splat.splat_utils import GSplatLoader
from splatplan.splatplan1 import SplatPlan


class SplatPlannerNode(Node):
    def __init__(self):
        super().__init__('splat_planner_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale = (1.0 / 0.1778384719584236) / 1.78

        # self.nerf = NeRF(config_path=Path(f"/f1tenth_splatnav/outputs/vicon_working/gemsplat/2025-04-19_192539/config.yml"),
        #     res_factor=None,
        #     test_mode="test",
        #     dataset_mode="val",
        #     device=self.device)
        
        # self.nerf.generate_point_cloud(use_bounding_box=True)

        # del self.nerf, self.env_pcd, self.env_attr

        # gc.collect()
        # torch.cuda.empty_cache()

        # print("Goal set")

        # Config path
        gsplat_config_path = Path("/f1tenth_splatnav/outputs/vicon_working/gemsplat/2025-04-19_192539/config.yml")

        # Load GSplat and Planner
        self.gsplat = GSplatLoader(gsplat_config_path, self.device)
        self.voxel_config = {
            'lower_bound': torch.tensor([-1, -1, -0.35], device=self.device),
            'upper_bound': torch.tensor([1, 1, -0.05], device=self.device),
            'resolution': torch.tensor([200, 200, 3], device=self.device),
        }
        self.robot_config = {'radius': 0.06}
        self.planner = SplatPlan(self.gsplat, self.robot_config, self.voxel_config, self.device)

        # ROS interfaces
        qos_transient = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.path_pub = self.create_publisher(Path_ros, '/splatnav_path', qos_transient)
        self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', qos_transient)
        # odom = '/ego_racecar/odom'
        odom = '/pf/pose/odom'
        self.odom_sub = self.create_subscription(Odometry, odom, self.odom_callback, 10)
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



    # def clicked_callback(self, msg: PointStamped):
    #     self.goal = torch.tensor([
    #         msg.point.x / self.scale,
    #         msg.point.y / self.scale,
    #         -0.35
    #     ], device=self.device)

    #     self.get_logger().info(f"Received goal: {self.goal.cpu().numpy()}")
    #     self.planned = False  # allow replanning on new goal

    def odom_callback(self, msg: Odometry):
        if self.goal is None:
            self.get_logger().info("Waiting for goal from RViz...")
            return

        # Get current pose
        start = torch.tensor([
            msg.pose.pose.position.x / self.scale,
            msg.pose.pose.position.y / self.scale,
            -0.35
        ], device=self.device)

        if self.planned and self.latest_start is not None:
            # If already planned, just republish latched topics
            self.path_msg.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(self.path_msg)
            self.publish_goal_marker(goal_pos=(self.goal[0].item() * self.scale, self.goal[1].item() * self.scale))
            return

        self.get_logger().info(f"Planning from: {start.cpu().numpy()} to {self.goal.cpu().numpy()}")

        output = self.planner.generate_path(start, self.goal)
        if output['path'] is None:
            self.get_logger().warn(f"Planning from: {start.cpu().numpy()} to {self.goal.cpu().numpy()} failed")
            self.goal = None
            return 

        # Convert to Path message
        self.path_msg = Path_ros()
        ## append the start pose to the path
        # Interpolate from current position to first waypoint
        start_np = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        first_wp = np.array([output['path'][0][0] * self.scale, output['path'][0][1] * self.scale])
        num_interp = int(np.linalg.norm(first_wp - start_np) / 0.1)  # spacing ~10cm

        interp_points = np.linspace(start_np, first_wp, num=num_interp + 1)

        for pt in interp_points:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            self.path_msg.poses.append(pose)

        
        self.path_msg.header.frame_id = 'map'
        self.path_msg.header.stamp = self.get_clock().now().to_msg()

        for pt in output['path']:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = pt[0] * self.scale
            pose.pose.position.y = pt[1] * self.scale
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            self.path_msg.poses.append(pose)

        self.path_pub.publish(self.path_msg)
        
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
    node = SplatPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
