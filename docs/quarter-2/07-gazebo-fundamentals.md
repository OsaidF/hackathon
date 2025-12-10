---
title: "Chapter 7: Gazebo Fundamentals"
sidebar_label: "7. Gazebo Fundamentals"
sidebar_position: 7
---

import { PythonCode } from '@site/src/components/CodeBlock';
import { BashCode } from '@site/src/components/CodeBlock';
import { ROS2Code } from '@site/src/components/CodeBlock';

# Chapter 7: Gazebo Fundamentals

## Building Complete Robotics Simulation Environments

Welcome to Chapter 7, where we dive deep into Gazebo - the leading open-source robotics simulation platform that bridges the gap between theoretical robotics concepts and practical implementation. Gazebo provides realistic physics, sensor simulation, and visualization capabilities that make it an indispensable tool for modern robotics development.

## 🎯 Chapter Learning Objectives

By the end of this chapter, you will be able to:

1. **Master Gazebo Architecture**: Understand core components and communication patterns
2. **Build Complete Worlds**: Design realistic simulation environments with proper physics
3. **Integrate Sensors and Actuators**: Implement realistic sensor simulation and robot control
4. **ROS 2 Integration**: Seamlessly connect Gazebo simulations with ROS 2 applications
5. **Optimize Performance**: Balance realism with computational efficiency for large-scale simulations

## 🏗️ Gazebo Architecture Overview

### Core Components

Gazebo consists of several interconnected components that work together to create realistic simulations:

#### 1. Gazebo Server (gzserver)
The physics engine and simulation backend:

<BashCode title="Starting Gazebo Server">
```bash
# Start Gazebo server with specific physics engine
gzserver --physics-engine bullet -s libgazebo_ros_init.so

# Run in verbose mode for debugging
gzserver --verbose -s libgazebo_ros_factory.so

# Start with custom world file
gzserver custom_world.sdf -s libgazebo_ros_api_plugin.so
```
</BashCode>

#### 2. Gazebo Client (gzclient)
The visualization and GUI component:

<BashCode title="Starting Gazebo Client">
```bash
# Start Gazebo client (GUI)
gzclient

# Connect to remote server
gzclient -g 192.168.1.100:11345

# Start with custom configuration
gzclient -c client_config.conf
```
</BashCode>

#### 3. Gazebo Plugins
Extensible modules for custom functionality:

<PythonCode title="Understanding Gazebo Plugin Architecture">
```python
# Example: World plugin structure in C++
/*
class WorldPluginTutorial : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr _world, sdf::ElementPtr _sdf)
  {
    // Plugin initialization
    this->world = _world;

    // Connect to update event
    this->updateConnection = gazebo::event::Events::ConnectWorldUpdateBegin(
        std::bind(&WorldPluginTutorial::OnUpdate, this));
  }

  void OnUpdate()
  {
    // Called every simulation step
    // Implement custom behavior here
  }

private:
  gazebo::physics::WorldPtr world;
  gazebo::event::ConnectionPtr updateConnection;
};

// Register plugin with Gazebo
GZ_REGISTER_WORLD_PLUGIN(WorldPluginTutorial)
*/
```
</PythonCode>

## 🌍 World Creation and Management

### SDF (Simulation Description Format)

SDF is the native format for describing Gazebo worlds and models:

<PythonCode title="Basic SDF World Structure">
```python
sdf_world_example = """
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="example_world">

    <!-- Physics engine configuration -->
    <physics name="1ms" type="bullet">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Gravity settings -->
    <gravity>0 0 -9.8066</gravity>

    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun for lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom model -->
    <model name="custom_robot">
      <pose>0 0 0.5 0 0 0</pose>

      <link name="base_link">
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <iyy>1.0</iyy>
            <izz>1.0</izz>
          </inertia>
        </inertial>

        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 0.5</size>
            </box>
          </geometry>
        </collision>

        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
"""
```
</PythonCode>

### Creating Complex Worlds

<BashCode title="Building Complex Simulation Worlds">
```bash
# Create new Gazebo world
gazebo --verbose -s libgazebo_ros_factory.so

# Use world editor for interactive design
gazebo --gui -e world_editor

# Generate world from existing models
gazebo --spawn-model.urdf my_robot.urdf -m my_robot -x 0 -y 0 -z 1

# Save current world to SDF
gz world -o my_custom_world.sdf
```
</BashCode>

<PythonCode title="Python Script for World Generation">
```python
import xml.etree.ElementTree as ET
import numpy as np

def create_warehouse_world(output_file="warehouse_world.sdf"):
    """Generate a warehouse simulation world"""

    # Create root SDF element
    sdf = ET.Element("sdf", version="1.7")
    world = ET.SubElement(sdf, "world", name="warehouse")

    # Add physics configuration
    physics = ET.SubElement(world, "physics", name="1ms", type="bullet")
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_factor").text = "1.0"

    # Add ground plane
    ground_include = ET.SubElement(world, "include")
    ET.SubElement(ground_include, "uri").text = "model://ground_plane"

    # Add lighting
    sun_include = ET.SubElement(world, "include")
    ET.SubElement(sun_include, "uri").text = "model://sun"

    # Create warehouse shelving units
    shelf_positions = [
        (-5, 0, 0), (-5, 3, 0), (-5, -3, 0),
        (5, 0, 0), (5, 3, 0), (5, -3, 0),
        (0, 5, 1.57), (0, -5, 1.57)
    ]

    for i, (x, y, yaw) in enumerate(shelf_positions):
        shelf_include = ET.SubElement(world, "include")
        ET.SubElement(shelf_include, "uri").text = "model://warehouse_shelf"
        pose = ET.SubElement(shelf_include, "pose")
        pose.text = f"{x} {y} 0 0 0 {yaw}"

        # Assign unique name
        ET.SubElement(shelf_include, "name").text = f"shelf_{i}"

    # Add obstacles
    obstacle_positions = [
        (2, 2, 0.5), (-2, 2, 0.5), (2, -2, 0.5), (-2, -2, 0.5)
    ]

    for i, (x, y, z) in enumerate(obstacle_positions):
        obstacle = ET.SubElement(world, "model", name=f"obstacle_{i}")
        ET.SubElement(obstacle, "pose").text = f"{x} {y} {z} 0 0 0"

        static_link = ET.SubElement(obstacle, "static").text = "true"

        link = ET.SubElement(obstacle, "link", name="link")

        # Collision geometry
        collision = ET.SubElement(link, "collision", name="collision")
        geometry = ET.SubElement(collision, "geometry")
        cylinder = ET.SubElement(geometry, "cylinder")
        ET.SubElement(cylinder, "radius").text = "0.2"
        ET.SubElement(cylinder, "length").text = "1.0"

        # Visual geometry
        visual = ET.SubElement(link, "visual", name="visual")
        visual_geometry = ET.SubElement(visual, "geometry")
        visual_cylinder = ET.SubElement(visual_geometry, "cylinder")
        ET.SubElement(visual_cylinder, "radius").text = "0.2"
        ET.SubElement(visual_cylinder, "length").text = "1.0"

        # Material
        material = ET.SubElement(visual, "material")
        ET.SubElement(material, "ambient").text = "0.8 0.2 0.2 1"
        ET.SubElement(material, "diffuse").text = "0.8 0.2 0.2 1"

    # Pretty print and save
    ET.indent(sdf, space="  ")
    tree = ET.ElementTree(sdf)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Warehouse world saved to {output_file}")

# Generate the world
create_warehouse_world()
```
</PythonCode>

## 🤖 Robot Models in Gazebo

### URDF to SDF Integration

Converting URDF models for Gazebo simulation:

<PythonCode title="URDF with Gazebo Extensions">
```python
urdf_with_gazebo = """
<?xml version="1.0"?>
<robot name="mobile_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="20.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0"
               iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>

    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo-specific extensions -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.5</mu1>
    <mu2>0.5</mu2>
    <kp>1000000.0</kp>
    <kd>1.0</kd>
    <minDepth>0.001</minDepth>
    <maxContacts>100</maxContacts>
  </gazebo>

  <!-- Differential drive wheels -->
  <link name="left_wheel">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0"
               iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>

    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 0" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>1000000.0</kp>
    <kd>1.0</kd>
    <minDepth>0.001</minDepth>
    <maxContacts>100</maxContacts>
  </gazebo>

  <!-- Right wheel (similar structure) -->
  <link name="right_wheel">
    <!-- Similar inertial, visual, collision properties -->
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 0" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
  </gazebo>

  <!-- Differential drive plugin -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <legacyMode>false</legacyMode>
      <alwaysOn>true</alwaysOn>
      <updateRate>100</updateRate>
      <leftJoint>left_wheel_joint</leftJoint>
      <rightJoint>right_wheel_joint</rightJoint>
      <wheelSeparation>0.4</wheelSeparation>
      <wheelDiameter>0.1</wheelDiameter>
      <torque>5</torque>
      <commandTopic>cmd_vel</commandTopic>
      <odometryTopic>odom</odometryTopic>
      <odometryFrame>odom</odometryFrame>
      <robotBaseFrame>base_link</robotBaseFrame>
    </plugin>
  </gazebo>

  <!-- Laser scanner sensor -->
  <link name="laser_scanner">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0"
               iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>

    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="laser_scanner_joint" type="fixed">
    <parent link="base_link"/>
    <child link="laser_scanner"/>
    <origin xyz="0.15 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- Laser sensor plugin -->
  <gazebo reference="laser_scanner">
    <sensor type="ray" name="laser_scanner_sensor">
      <pose>0 0 0 0 0 0</pose>
      <visualize>false</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="gazebo_ros_laser" filename="libgazebo_ros_laser.so">
        <topicName>/scan</topicName>
        <frameName>laser_scanner</frameName>
      </plugin>
    </sensor>
  </gazebo>
</robot>
"""
```
</PythonCode>

### Model Spawning and Management

<BashCode title="Spawning and Managing Robot Models">
```bash
# Spawn URDF model in Gazebo
ros2 run gazebo_ros spawn_entity.py -urdf -file robot.urdf -entity my_robot

# Spawn with initial pose
ros2 run gazebo_ros spawn_entity.py \
  -urdf -file mobile_robot.urdf \
  -entity mobile_robot \
  -x 2.0 -y 1.0 -z 0.5 \
  -R 0 -P 0 -Y 1.57

# Delete model from simulation
rosservice call /delete_entity '{name: "my_robot"}'

# Get model state
ros2 topic echo /gazebo/model_states

# List all models
rosservice call /get_world_properties '{}'

# Set model pose
rosservice call /set_entity_state '{entity_name: "my_robot", state: {pose: {position: {x: 1, y: 1, z: 0.5}, orientation: {x: 0, y: 0, z: 0, w: 1}}}}'
```
</BashCode>

## 📡 Sensor Simulation

### Camera Sensors

<PythonCode title="Camera Sensor Configuration">
```python
camera_sensor_config = """
<gazebo reference="camera_link">
  <sensor type="camera" name="camera_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>30</update_rate>
    <camera name="camera_head">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>1280</width>
        <height>720</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <image_topic_name>/camera/image_raw</image_topic_name>
      <camera_info_topic_name>/camera/camera_info</camera_info_topic_name>
      <frame_name>camera_link</frame_name>
      <hack_baseline>0.07</hack_baseline>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
"""
```
</PythonCode>

### Depth Cameras and 3D Sensing

<PythonCode title="Depth Camera Sensor">
```python
depth_camera_config = """
<gazebo reference="depth_camera_link">
  <sensor type="depth" name="depth_camera_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>30</update_rate>
    <camera name="depth_camera_head">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>L8</format>
      </image>
      <depth>
        <near>0.1</near>
        <far>10.0</far>
      </depth>
      <clip>
        <near>0.1</near>
        <far>100.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_depth_camera.so">
      <image_topic_name>/depth_camera/image_raw</image_topic_name>
      <camera_info_topic_name>/depth_camera/camera_info</camera_info_topic_name>
      <depth_image_topic_name>/depth_camera/depth/image_raw</depth_image_topic_name>
      <depth_image_camera_info_topic_name>/depth_camera/depth/camera_info</depth_image_camera_info_topic_name>
      <point_cloud_topic_name>/depth_camera/points</point_cloud_topic_name>
      <frame_name>depth_camera_link</frame_name>
      <point_cloud_cutoff>0.5</point_cloud_cutoff>
    </plugin>
  </sensor>
</gazebo>
"""
```
</PythonCode>

### IMU and Inertial Sensors

<PythonCode title="IMU Sensor Configuration">
```python
imu_sensor_config = """
<gazebo reference="imu_link">
  <sensor type="imu" name="imu_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>false</visualize>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0002</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0002</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0002</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
      <topicName>/imu/data</topicName>
      <bodyName>imu_link</bodyName>
      <updateRateHZ>100.0</updateRateHZ>
      <gaussianNoise>0.0</gaussianNoise>
      <xyzOffset>0 0 0</xyzOffset>
      <rpyOffset>0 0 0</rpyOffset>
    </plugin>
  </sensor>
</gazebo>
"""
```
</PythonCode>

## 🔌 ROS 2 Integration

### Launch Files for Gazebo Simulations

<PythonCode title="Gazebo Launch File with ROS 2">
```python
launch_file_content = """
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Declare launch arguments
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value='empty.world',
        description='Gazebo world file name'
    )

    robot_urdf_arg = DeclareLaunchArgument(
        'robot_urdf',
        default_value='mobile_robot.urdf',
        description='Robot URDF file name'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # Get package directories
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_share = get_package_share_directory('robot_simulation')

    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': LaunchConfiguration('world_file')}.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'robot_description': Command(['xacro ', LaunchConfiguration('robot_urdf')])}]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'mobile_robot',
                   '-topic', 'robot_description',
                   '-x', '0', '-y', '0', '-z', '0.5'],
        output='screen'
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # RViz2 visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'config', 'robot.rviz')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Teleop control node
    teleop_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        output='screen',
        prefix='xterm -e'
    )

    return LaunchDescription([
        world_file_arg,
        robot_urdf_arg,
        use_sim_time_arg,
        gazebo_launch,
        robot_state_publisher,
        spawn_entity,
        joint_state_publisher,
        rviz_node,
        teleop_node
    ])
"""
```
</PythonCode>

### ROS 2 Topics and Services

<ROS2Code title="Common Gazebo ROS 2 Topics">
```bash
# Model state and control
/gazebo/model_states          # Model poses and velocities
/gazebo/set_model_state       # Set model pose service
/gazebo/delete_entity         # Delete model service
/gazebo/spawn_entity          # Spawn model service

# Joint control
/joint_states                 # Joint positions and velocities
/joint_state_broadcaster      # Joint state publisher
/controller_manager           # Controller management

# Sensor data
/camera/image_raw            # Camera images
/camera/camera_info          # Camera calibration
/laser/scan                  # Laser scanner data
/depth_camera/points         # Point cloud data
/imu/data                    # IMU measurements

# Robot control
/cmd_vel                     # Velocity commands
/odom                        # Odometry data
/tf                          # Transform tree
/tf_static                   # Static transforms
```
</ROS2Code>

## 🎮 Advanced Gazebo Features

### Multi-Robot Simulations

<PythonCode title="Multi-Robot Simulation Setup">
```python
def launch_multi_robot_simulation(num_robots=3):
    """Launch simulation with multiple robots"""

    import xml.etree.ElementTree as ET

    for i in range(num_robots):
        # Create robot model with unique namespace
        robot_model = create_robot_model(namespace=f"robot_{i}")

        # Set different initial positions
        x_pos = i * 2.0
        y_pos = 0.0

        # Generate spawn command
        spawn_command = f"""
        ros2 run gazebo_ros spawn_entity.py \\
            -urdf -file robot.urdf \\
            -entity robot_{i} \\
            -x {x_pos} -y {y_pos} -z 0.5 \\
            -ros-namespace robot_{i}
        """

        print(f"Spawning robot_{i} at position ({x_pos}, {y_pos}, 0.5)")
        print(spawn_command)

def create_robot_model(namespace="robot"):
    """Create URDF model with specific namespace"""
    urdf_template = f"""
    <?xml version="1.0"?>
    <robot name="{namespace}" xmlns:xacro="http://www.ros.org/wiki/xacro">

      <!-- Add namespace to all links and joints -->
      <link name="{namespace}/base_link">
        <!-- Link definition -->
      </link>

      <!-- Gazebo plugin with namespace -->
      <gazebo>
        <plugin name="diff_drive_{namespace}" filename="libgazebo_ros_diff_drive.so">
          <ros>
            <namespace>/{namespace}</namespace>
          </ros>
          <commandTopic>cmd_vel</commandTopic>
          <odometryTopic>odom</odometryTopic>
          <odometryFrame>odom</odometryFrame>
          <robotBaseFrame>base_link</robotBaseFrame>
        </plugin>
      </gazebo>

      <!-- Sensors with namespace -->
      <gazebo reference="laser_scanner">
        <sensor type="ray" name="laser_sensor">
          <plugin name="gazebo_ros_laser_{namespace}" filename="libgazebo_ros_laser.so">
            <ros>
              <namespace>/{namespace}</namespace>
            </ros>
            <topicName>scan</topicName>
            <frameName>laser_scanner</frameName>
          </plugin>
        </sensor>
      </gazebo>

    </robot>
    """

    return urdf_template
```
</PythonCode>

### Custom Plugin Development

<PythonCode title="Custom Gazebo Plugin Template">
```python
"""
Custom Gazebo World Plugin for Autonomous Navigation Challenges
"""

# This would be implemented in C++, here's the Python structure

cpp_plugin_template = """
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>

#include <ignition/math/Pose3.hh>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>

namespace gazebo
{
  class NavigationChallenge : public WorldPlugin
  {
    public:
      void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
      {
        this->world = _world;

        // Initialize ROS node
        if (!ros::isInitialized())
        {
          int argc = 0;
          char **argv = NULL;
          ros::init(argc, argv, "gazebo_navigation_challenge",
                    ros::init_options::NoSigintHandler);
        }

        this->rosNode.reset(new ros::NodeHandle("gazebo_navigation_challenge"));

        // Create publishers
        goalPublisher = this->rosNode->advertise<geometry_msgs::PoseStamped>(
          "/navigation/goal", 1);

        statusPublisher = this->rosNode->advertise<std_msgs::String>(
          "/navigation/status", 1);

        // Create subscribers
        goalReachedSubscriber = this->rosNode->subscribe(
          "/navigation/goal_reached", 1,
          &NavigationChallenge::OnGoalReached, this);

        // Connect to world update event
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&NavigationChallenge::OnUpdate, this));

        // Initialize challenge waypoints
        InitializeWaypoints();

        ROS_INFO("Navigation Challenge plugin loaded!");
      }

    private:
      void InitializeWaypoints()
      {
        waypoints.clear();

        // Define challenge waypoints
        waypoints.push_back(ignition::math::Pose3d(5, 0, 0, 0, 0, 0));
        waypoints.push_back(ignition::math::Pose3d(5, 5, 0, 0, 0, 1.57));
        waypoints.push_back(ignition::math::Pose3d(0, 5, 0, 0, 0, 3.14));
        waypoints.push_back(ignition::math::Pose3d(0, 0, 0, 0, 0, -1.57));

        currentWaypoint = 0;
        challengeStartTime = this->world->SimTime();
      }

      void OnUpdate()
      {
        // Update challenge logic
        if (!challengeCompleted)
        {
          CheckTimeLimit();
          UpdateChallengeStatus();
        }
      }

      void OnGoalReached(const std_msgs::String::ConstPtr& msg)
      {
        ROS_INFO("Goal reached: %s", msg->data.c_str());

        currentWaypoint++;

        if (currentWaypoint >= waypoints.size())
        {
          challengeCompleted = true;
          PublishChallengeCompleted();
        }
        else
        {
          PublishNextWaypoint();
        }
      }

      void PublishNextWaypoint()
      {
        if (currentWaypoint < waypoints.size())
        {
          geometry_msgs::PoseStamped goal;
          goal.header.stamp = ros::Time::now();
          goal.header.frame_id = "world";

          ignition::math::Pose3d waypoint = waypoints[currentWaypoint];
          goal.pose.position.x = waypoint.Pos().X();
          goal.pose.position.y = waypoint.Pos().Y();
          goal.pose.position.z = waypoint.Pos().Z();

          goal.pose.orientation.w = waypoint.Rot().W();
          goal.pose.orientation.x = waypoint.Rot().X();
          goal.pose.orientation.y = waypoint.Rot().Y();
          goal.pose.orientation.z = waypoint.Rot().Z();

          goalPublisher.publish(goal);

          ROS_INFO("Published waypoint %d at (%.2f, %.2f)",
                   currentWaypoint, waypoint.Pos().X(), waypoint.Pos().Y());
        }
      }

      void CheckTimeLimit()
      {
        gazebo::common::Time currentTime = this->world->SimTime();
        gazebo::common::Time elapsedTime = currentTime - challengeStartTime;

        if (elapsedTime.Double() > timeLimit)
        {
          challengeCompleted = true;
          PublishTimeLimitExceeded();
        }
      }

      void UpdateChallengeStatus()
      {
        gazebo::common::Time currentTime = this->world->SimTime();
        gazebo::common::Time elapsedTime = currentTime - challengeStartTime;

        std_msgs::String status;
        status.data = "Waypoint: " + std::to_string(currentWaypoint) +
                     "/" + std::to_string(waypoints.size()) +
                     " | Time: " + std::to_string(elapsedTime.Double()) + "s";

        statusPublisher.publish(status);
      }

    private:
      physics::WorldPtr world;
      event::ConnectionPtr updateConnection;

      std::unique_ptr<ros::NodeHandle> rosNode;
      ros::Publisher goalPublisher;
      ros::Publisher statusPublisher;
      ros::Subscriber goalReachedSubscriber;

      std::vector<ignition::math::Pose3d> waypoints;
      size_t currentWaypoint;

      gazebo::common::Time challengeStartTime;
      double timeLimit = 300.0;  // 5 minutes
      bool challengeCompleted = false;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_WORLD_PLUGIN(NavigationChallenge)
}
"""
```
</PythonCode>

## 📊 Performance Optimization

### Physics Tuning

<PythonCode title="Gazebo Performance Optimization Settings">
```python
performance_config = """
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="optimized_world">

    <!-- Optimized physics settings -->
    <physics name="default" type="bullet">
      <max_step_size>0.002</max_step_size>           <!-- 500 Hz -->
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>500</real_time_update_rate>

      <!-- Bullet specific optimizations -->
      <bullet>
        <solver_type>si</solver_type>
        <split_impulse>true</split_impulse>
        <split_impulse_penetration_threshold>-0.01</split_impulse_penetration_threshold>
        <allowed_ccd_penalty>0.01</allowed_ccd_penalty>
        <use_split_impulse>true</use_split_impulse>
        <itr_type>si</itr_type>
        <num_solver_iterations>20</num_solver_iterations>
        <s Sor>1.3</sor>
        <contact_surface_layer>0.001</contact_surface_layer>
        <ERP>0.2</ERP>
        <contactERP>0.2</contactERP>
        <frictionERP>0.2</frictionERP>
        <constraint_force mixing>0.0</constraint_force mixing>
      </bullet>
    </physics>

    <!-- Performance monitoring -->
    <gui>
      <camera name="user_camera">
        <pose>-6 -6 6 0 0.275643 2.35619</pose>
      </camera>
    </gui>

    <!-- Simplified models for performance -->
    <model name="simple_ground">
      <static>true</static>
      <link name="ground_link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
      </link>
    </model>
  </world>
</sdf>
"""
```
</PythonCode>

### Level of Detail (LOD) Management

<PythonCode title="Implementing LOD for Complex Models">
```python
def create_lod_model_config():
    """Create LOD configuration for performance optimization"""

    lod_config = """
    <model name="detailed_robot">
      <!-- High detail LOD (close camera) -->
      <model name="lod_high">
        <static>false</static>
        <pose>0 0 0 0 0 0</pose>

        <!-- Detailed collision and visual -->
        <link name="base_link_high">
          <inertial>
            <mass value="10.0"/>
            <!-- Full inertia tensor -->
          </inertial>

          <!-- Detailed collision geometry -->
          <collision name="detailed_collision">
            <geometry>
              <mesh>
                <uri>model://robot/meshes/detailed_base.dae</uri>
              </mesh>
            </geometry>
          </collision>

          <!-- High resolution visual -->
          <visual name="detailed_visual">
            <geometry>
              <mesh>
                <uri>model://robot/meshes/detailed_base.dae</uri>
              </mesh>
            </visual>
            <material>
              <script>
                <uri>file://media/materials/scripts/gazebo.material</uri>
                <name>Gazebo/Chrome</name>
              </script>
            </material>
          </visual>
        </link>
      </model>

      <!-- Medium detail LOD (medium camera distance) -->
      <model name="lod_medium">
        <static>false</static>
        <pose>0 0 0 0 0 0</pose>

        <!-- Simplified geometry -->
        <link name="base_link_medium">
          <inertial>
            <mass value="10.0"/>
            <!-- Simplified inertia -->
          </inertial>

          <!-- Simplified collision -->
          <collision name="simplified_collision">
            <geometry>
              <box>
                <size>1 1 0.5</size>
              </box>
            </geometry>
          </collision>

          <!-- Medium resolution visual -->
          <visual name="medium_visual">
            <geometry>
              <box>
                <size>1 1 0.5</size>
              </box>
            </visual>
            <material>
              <ambient>0.7 0.7 0.7 1</ambient>
              <diffuse>0.7 0.7 0.7 1</diffuse>
            </material>
          </visual>
        </link>
      </model>

      <!-- Low detail LOD (far camera) -->
      <model name="lod_low">
        <static>false</static>
        <pose>0 0 0 0 0 0</pose>

        <!-- Minimal geometry -->
        <link name="base_link_low">
          <inertial>
            <mass value="10.0"/>
          </inertial>

          <!-- Box collision only -->
          <collision name="box_collision">
            <geometry>
              <box>
                <size>1 1 0.5</size>
              </box>
            </geometry>
          </collision>

          <!-- Simple visual -->
          <visual name="simple_visual">
            <geometry>
              <box>
                <size>1 1 0.5</size>
              </box>
            </visual>
            <material>
              <ambient>0.5 0.5 0.5 1</ambient>
            </material>
          </visual>
        </link>
      </model>

      <!-- LOD switching plugin -->
      <plugin name="lod_switcher" filename="libgazebo_ros_lod.so">
        <high_distance>5.0</high_distance>
        <medium_distance>15.0</medium_distance>
        <low_distance>50.0</low_distance>
        <high_model>lod_high</high_model>
        <medium_model>lod_medium</medium_model>
        <low_model>lod_low</low_model>
      </plugin>
    </model>
    """

    return lod_config
```
</PythonCode>

## 🎯 Chapter Project: Complete Mobile Robot Simulation

### Project Overview

Create a comprehensive mobile robot simulation that demonstrates all Gazebo concepts covered:

<PythonCode title="Complete Mobile Robot Launch System">
```python
#!/usr/bin/env python3

import os
import tempfile
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    # Declare arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='warehouse.world',
        description='Gazebo world file'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='warehouse_robot',
        description='Robot name'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode'
    )

    # Package paths
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_share = FindPackageShare('robot_simulation')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gazebo.launch.py']),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'headless': LaunchConfiguration('headless'),
            'verbose': 'true'
        }.items()
    )

    # Robot description
    robot_description_content = Command([
        PathJoinSubstitution([FindPackageShare('xacro'), 'xacro']),
        ' ',
        PathJoinSubstitution([pkg_share, 'urdf', 'warehouse_robot.urdf.xacro'])
    ])

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': robot_description_content
        }]
    )

    # Spawn robot
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', LaunchConfiguration('robot_name'),
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.5',
            '-robot_namespace', LaunchConfiguration('robot_name')
        ],
        output='screen'
    )

    # Navigation stack
    navigation_stack = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[PathJoinSubstitution([pkg_share, 'config', 'nav2_params.yaml'])],
        remappings=[
            ('scan', '/laser/scan'),
            ('map', '/map'),
            ('odom', '/odom')
        ]
    )

    # Path planning
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[PathJoinSubstitution([pkg_share, 'config', 'nav2_params.yaml'])]
    )

    # Controller
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[PathJoinSubstitution([pkg_share, 'config', 'nav2_params.yaml'])]
    )

    # Map server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': PathJoinSubstitution([pkg_share, 'maps', 'warehouse.yaml']),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )

    # Lifecycle manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'autostart': True,
            'node_names': ['map_server', 'amcl', 'planner_server', 'controller_server']
        }]
    )

    # RViz2
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'warehouse_navigation.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Task monitoring
    task_monitor = Node(
        package='robot_simulation',
        executable='task_monitor',
        name='task_monitor',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )

    return LaunchDescription([
        world_arg,
        robot_name_arg,
        use_sim_time_arg,
        headless_arg,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        map_server,
        amcl,
        planner_server,
        controller_server,
        lifecycle_manager,
        rviz_node,
        task_monitor
    ])

if __name__ == '__main__':
    generate_launch_description()
```
</PythonCode>

## 📋 Chapter Summary

### Key Concepts Covered

1. **Gazebo Architecture**: Server-client model and plugin system
2. **World Creation**: SDF format and complex environment design
3. **Robot Models**: URDF integration and Gazebo extensions
4. **Sensor Simulation**: Cameras, laser scanners, IMU, and depth sensors
5. **ROS 2 Integration**: Launch files, topics, and services
6. **Advanced Features**: Multi-robot simulations and custom plugins
7. **Performance Optimization**: Physics tuning and LOD management

### Practical Skills Acquired

- ✅ Build complex Gazebo worlds with realistic physics
- ✅ Integrate sensor models with proper noise characteristics
- ✅ Develop custom Gazebo plugins for specialized functionality
- ✅ Optimize simulation performance for large-scale scenarios
- ✅ Create complete ROS 2 + Gazebo launch systems

### Next Steps

This Gazebo foundation prepares you for **Chapter 8: Unity Robotics**, where you'll explore advanced visualization and simulation capabilities using Unity's powerful rendering engine. You'll learn how to:

- Create photorealistic robot simulations
- Implement advanced rendering effects
- Build interactive training environments
- Integrate Unity simulations with ROS 2

---

## 🤔 Chapter Reflection

1. **Architecture Understanding**: How does Gazebo's plugin system enable extensibility for specialized robotics applications?
2. **Performance Trade-offs**: When should you prioritize simulation realism over computational efficiency?
3. **Integration Strategy**: What are the key considerations when designing ROS 2 + Gazebo systems?
4. **Application**: How can you extend these Gazebo concepts for your specific robotics research or development needs?

---

**[← Back to Quarter 2 Overview](index.md) | [Continue to Chapter 8: Unity Robotics →](08-unity-robotics.md)**