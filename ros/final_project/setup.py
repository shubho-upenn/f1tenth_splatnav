from setuptools import setup

package_name = 'final_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'publish_pcl',
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
    description='Publishes pointcloud from ply file to ROS2 topic',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publish_pcl = publish_pcl:main',
        ],
    },
)
