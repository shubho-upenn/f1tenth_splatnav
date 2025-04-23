# import sys
# import os

# # Add the absolute path to f1tenth_splatnav to import splat modules
# splatnav_path = os.path.abspath("/f1tenth_splatnav")
# sys.path.insert(0, splatnav_path)


# import rclpy
# from rclpy.node import Node
# from nav_msgs.msg import Odometry
# from nav_msgs.msg import Path as Path_ros
# from geometry_msgs.msg import PoseStamped
# import torch
# from pathlib import Path
# import numpy as np
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
# from visualization_msgs.msg import Marker

# from splat.splat_utils import GSplatLoader
# from splatplan.splatplan1 import SplatPlan


# class SplatPlannerNode(Node):
#     def __init__(self):
#         super().__init__('splat_planner_node')

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         # self.goal = torch.tensor([-0.5, 0.5, -0.35], device=self.device)  # hardcoded goal
#         self.goal = None
        
#         # Config path
#         gsplat_config_path = Path("/f1tenth_splatnav/outputs/vicon_small/splatfacto/2025-04-22_005347/config.yml")

#         # Load GSplat + Planner
#         self.gsplat = GSplatLoader(gsplat_config_path, self.device)
#         self.voxel_config = {
#             'lower_bound': torch.tensor([-1, -1, -0.4], device=self.device),
#             'upper_bound': torch.tensor([1, 1, -0.05], device=self.device),
#             'resolution': torch.tensor([100, 100, 5], device=self.device),
#         }
#         self.robot_config = {'radius': 0.06}
#         self.scale = 1.0/0.1778384719584236
#         self.planner = SplatPlan(self.gsplat, self.robot_config, self.voxel_config, self.device)

#         self.path_pub = self.create_publisher(Path_ros, '/splatnav_path', 10)
#         self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', 10)
#         self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

#         self.planned = False  # Plan only once (set True after first plan)

#     def odom_callback(self, msg: Odometry):
#         if self.planned:
#             self.path_msg.header.stamp = self.get_clock().now().to_msg()
#             self.path_pub.publish(self.path_msg)
#             self.publish_goal_marker(goal_pos=(self.goal[0].item(), self.goal[1].item()))
#             return  # prevent repeated planning

#         if self.goal is None:
#             self.get_logger().info(f"Waiting for goal")
#             return

#         scale = self.scale

#         # Get current pose
#         start = torch.tensor([
#             msg.pose.pose.position.x / scale,
#             msg.pose.pose.position.y / scale,
#             -0.35  # fixed Z slice
#         ], device=self.device)

#         self.get_logger().info(f"Planning from start: {start.cpu().numpy()}")

#         # Plan
#         output = self.planner.generate_path(start, self.goal)

#         # Convert to Path msg
#         self.path_msg = Path_ros()
#         self.path_msg.header.frame_id = 'map'
#         self.path_msg.header.stamp = self.get_clock().now().to_msg()

#         for pt in output['path']:
#             pose = PoseStamped()
#             pose.header.frame_id = 'map'
#             pose.pose.position.x = pt[0] * scale
#             pose.pose.position.y = pt[1] * scale
#             pose.pose.position.z = 0.0  # Z ignored for 2D controller
#             pose.pose.orientation.w = 1.0
#             self.path_msg.poses.append(pose)

#         self.path_pub.publish(self.path_msg)
#         self.publish_goal_marker(goal_pos=(self.goal[0].item() * scale, self.goal[1].item() * scale))
#         self.get_logger().info("Published RRT path to /splatnav_path")
#         self.planned = True

#     def publish_goal_marker(self, goal_pos):
#         marker = Marker()
#         marker.header.frame_id = "map"
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "goal"
#         marker.id = 0
#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD

#         marker.pose.position.x = goal_pos[0]
#         marker.pose.position.y = goal_pos[1]
#         marker.pose.position.z = 0.0
#         marker.pose.orientation.w = 1.0

#         marker.scale.x = 0.3
#         marker.scale.y = 0.3
#         marker.scale.z = 0.3

#         marker.color.r = 1.0
#         marker.color.g = 0.0
#         marker.color.b = 0.0
#         marker.color.a = 1.0

#         marker.lifetime.sec = 0  # forever

#         self.goal_marker_pub.publish(marker)



# def main(args=None):
#     rclpy.init(args=args)
#     node = SplatPlannerNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()


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

import torch
from pathlib import Path
import numpy as np

from splat.splat_utils import GSplatLoader
from splatplan.splatplan1 import SplatPlan


class SplatPlannerNode(Node):
    def __init__(self):
        super().__init__('splat_planner_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.goal = None
        self.scale = 1.0 / 0.1778384719584236

        # Config path
        gsplat_config_path = Path("/f1tenth_splatnav/outputs/vicon_small/splatfacto/2025-04-22_005347/config.yml")

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

        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.clicked_sub = self.create_subscription(PointStamped, '/clicked_point', self.clicked_callback, 10)

        self.path_msg = None
        self.latest_start = None
        self.planned = False

    def clicked_callback(self, msg: PointStamped):
        self.goal = torch.tensor([
            msg.point.x / self.scale,
            msg.point.y / self.scale,
            -0.35
        ], device=self.device)

        self.get_logger().info(f"Received goal: {self.goal.cpu().numpy()}")
        self.planned = False  # allow replanning on new goal

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
        self.publish_goal_marker(goal_pos=(self.goal[0].item() * self.scale, self.goal[1].item() * self.scale))
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
