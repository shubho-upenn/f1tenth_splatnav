import open3d as o3d
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import struct

class PointCloudPublisher(Node):
    def __init__(self, ply_path):
        super().__init__('pointcloud_publisher')

        # QoS: Transient local to keep message alive for new subscribers
        qos_profile = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.publisher = self.create_publisher(PointCloud2, '/gaussian_splat_pc', qos_profile)

        # Load and filter point cloud
        self.pcd = o3d.io.read_point_cloud(ply_path)
        all_points = np.asarray(self.pcd.points)
        all_colors = np.asarray(self.pcd.colors)

        # Filter within bounding box
        bbox_min = np.array([-1.0, -1.0, -0.4])
        bbox_max = np.array([ 1.0,  1.0, -0.05])
        mask = np.all((all_points >= bbox_min) & (all_points <= bbox_max), axis=1)
        points = all_points[mask]
        colors = all_colors[mask]

        # Apply Nerfstudio real-world transform
        transform = np.array([
            [0.9982146620750427, 0.05937574803829193, 0.006485733203589916, -0.05597333237528801],
            [0.006485733203589916, -0.21569545567035675, 0.9764391183853149, -0.04808540642261505],
            [0.05937574803829193, -0.9746537804603577, -0.21569545567035675, 0.038161471486091614]
        ])
        scale = (1.0/0.1778384719584236)/1.78
        R = transform[:, :3]
        t = transform[:, 3]

        # points = (R @ (scale * points.T)).T + t
        points[:, 2] = points[:, 2] + 0.4
        points = scale * points

        # map_origin_xy = np.array([-51.225, -51.225])  # from map.yaml
        # pointcloud_offset = np.array([0.0, 0.0])       # tune this manually if needed

        # points[:, 0] += map_origin_xy[0] + pointcloud_offset[0]  # X
        # points[:, 1] += map_origin_xy[1] + pointcloud_offset[1]  # Y



        self.points = points
        self.colors = colors

        self.get_logger().info(f"Transformed and publishing {len(self.points)} points.")
        self.publish_once()
    '''
    def publish_once(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_data = []
        for point, color in zip(self.points, self.colors):
            r, g, b = (color * 255).astype(np.uint8)
            rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
            cloud_data.append([point[0], point[1], point[2], rgb])

        msg = pc2.create_cloud(header, fields, cloud_data)
        self.publisher.publish(msg)
        self.get_logger().info("PointCloud2 message published once.")
    '''

    def publish_once(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Vectorized RGB to packed float conversion
        rgb_uint8 = (self.colors * 255).astype(np.uint8)
        rgb_packed = (rgb_uint8[:, 0].astype(np.uint32) << 16) | \
                     (rgb_uint8[:, 1].astype(np.uint32) << 8)  | \
                      rgb_uint8[:, 2].astype(np.uint32)
        rgb_floats = rgb_packed.view(np.float32)

        # Stack x, y, z, rgb into a single Nx4 array
        cloud_data = np.hstack((self.points, rgb_floats.reshape(-1, 1)))

        msg = pc2.create_cloud(header, fields, cloud_data)
        self.publisher.publish(msg)
        self.get_logger().info("PointCloud2 message published once.")


def main():
    rclpy.init()
    node = PointCloudPublisher('/home/praks/f1tenth_splatnav/outputs/vicon_working/mesh.ply')
    rclpy.spin(node)

if __name__ == '__main__':
    main()
