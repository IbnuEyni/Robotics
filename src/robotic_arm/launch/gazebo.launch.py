import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    package_share_dir = get_package_share_directory("robotic_arm")
    urdf_file = os.path.join(package_share_dir, "urdf", "robotic_arm.urdf")
    world_path = os.path.join(package_share_dir, "world", "my_world.sdf")
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', world_path],
                output="screen",
            ),
            
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=["-entity","robotic_arm","-b","-file", urdf_file,],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                arguments=[urdf_file],
            ),
            

        ]
    )