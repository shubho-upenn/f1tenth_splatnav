from setuptools import setup

package_name = 'final_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'publish_pcl',
        'splatplan_ros', 
        '2d_splatplan_ros',
        'semantic_splatplan_ros',
        'header',
        'semantic_2d_plan_ros',
        'semantic_2d_ackermann_plan', 
        '2d_ackermann_plan'

    ],
    package_dir={'': 'scripts'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shubho',
    maintainer_email='ssaditya@seas.upenn.edu',
    description='Publishes pointcloud from ply file to ROS2 topic and path from GSplat planner',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publish_pcl = publish_pcl:main',
            'splatplan_ros = splatplan_ros:main',  # <-- Entry point for new node
            '2d_splatplan_ros = 2d_splatplan_ros:main',
            'semantic_splatplan_ros = semantic_splatplan_ros:main',
            'semantic_2d_plan_ros= semantic_2d_plan_ros:main',
            'semantic_2d_ackermann_plan = semantic_2d_ackermann_plan:main',
            '2d_ackermann_plan = 2d_ackermann_plan:main'
        ],
    },
)
