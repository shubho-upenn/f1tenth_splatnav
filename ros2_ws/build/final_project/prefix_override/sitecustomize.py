import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/f1tenth_splatnav/ros2_ws/install/final_project'
