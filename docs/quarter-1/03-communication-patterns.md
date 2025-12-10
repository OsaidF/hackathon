---
title: "Chapter 3: Communication Patterns"
sidebar_label: "3. Communication Patterns"
sidebar_position: 3
---

# Chapter 3: Communication Patterns

## The Language of Robotic Systems

In robotics, communication patterns are the fundamental protocols that enable different components to exchange information effectively. Just as human language has different modes (conversation, requests, stories), ROS 2 provides distinct communication patterns suited for different types of robotic interactions and data flows.

## 🔄 Publish-Subscribe Pattern

### Concept Overview

The publish-subscribe (pub-sub) pattern is the most widely used communication pattern in robotics. It enables one-to-many data distribution where publishers send data to named topics without knowing who, if anyone, is receiving it.

### Anatomy of Pub-Sub

```
Publisher A                 Topic: /laser_scan                 Subscriber X
    │                                                            ▲
    └───► [Message Data] ──────────────────────────────────────┘
         └─────────────────────► Subscriber Y
         └─────────────────────────────► Subscriber Z
```

### Real-World Robotics Example

```python
# Laser Scanner Publisher
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import random

class LaserScannerPublisher(Node):
    def __init__(self):
        super().__init__('laser_scanner')

        # Publisher with high-frequency data
        self.publisher_ = self.create_publisher(
            LaserScan, 'laser_scan', 10)

        # High-frequency timer (10 Hz)
        self.timer_ = self.create_timer(0.1, self.publish_scan)
        self.angle_min = -math.pi
        self.angle_max = math.pi
        self.angle_increment = math.pi / 180  # 1 degree
        self.ranges_size = int((self.angle_max - self.angle_min) / self.angle_increment) + 1

        self.get_logger().info('Laser Scanner Publisher started')

    def publish_scan(self):
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'laser_link'

        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.1

        scan.range_min = 0.1
        scan.range_max = 10.0

        # Simulate laser scan data with obstacles
        ranges = []
        for i in range(self.ranges_size):
            angle = self.angle_min + i * self.angle_increment
            # Simulate obstacles at certain angles
            if -0.5 < angle < 0.5:  # Front obstacle
                distance = 2.0 + random.uniform(-0.1, 0.1)
            elif 1.5 < angle < 2.0:  # Side obstacle
                distance = 1.5 + random.uniform(-0.1, 0.1)
            else:
                distance = 5.0 + random.uniform(-0.5, 0.5)
            ranges.append(min(distance, 10.0))

        scan.ranges = ranges

        self.publisher_.publish(scan)

# Multiple Subscribers with Different Processing
class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.subscription = self.create_subscription(
            LaserScan, 'laser_scan', self.detect_obstacles, 10)

        self.obstacle_publisher = self.create_publisher(
            PointCloud2, 'obstacles', 10)

        self.get_logger().info('Obstacle Detector started')

    def detect_obstacles(self, msg):
        """Process laser scan to detect obstacles"""
        obstacles = []

        for i, distance in enumerate(msg.ranges):
            if distance < msg.range_max:
                angle = msg.angle_min + i * msg.angle_increment

                # Convert polar to Cartesian coordinates
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                z = 0.0

                obstacles.append([x, y, z])

        self.get_logger().info(f'Detected {len(obstacles)} obstacle points')
        # Publish obstacle point cloud...

class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')
        self.subscription = self.create_subscription(
            LaserScan, 'laser_scan', self.update_map, 10)

        self.map = {}  # Simple occupancy grid

        self.get_logger().info('Mapping Node started')

    def update_map(self, msg):
        """Update occupancy map with laser scan data"""
        for i, distance in enumerate(msg.ranges):
            if distance < msg.range_max:
                angle = msg.angle_min + i * msg.angle_increment

                # Convert to grid coordinates
                x = int(distance * math.cos(angle) * 10)  # 0.1m resolution
                y = int(distance * math.sin(angle) * 10)

                # Update map
                grid_key = (x, y)
                self.map[grid_key] = self.map.get(grid_key, 0) + 1

        self.get_logger().info(f'Map updated: {len(self.map)} occupied cells')
```

### Advanced Pub-Sub Features

#### Quality of Service (QoS) Profiles

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

# Different QoS for different use cases

# Real-time control data - loss acceptable, low latency
realtime_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)

# Critical safety data - must be reliable
safety_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_ALL
)

# Apply QoS to publisher
self.safety_publisher = self.create_publisher(
    EmergencyStop, 'emergency_stop', safety_qos)
```

#### Intra-Process Communication

```python
# Zero-copy communication for same-process nodes
self.publisher = self.create_publisher(
    LargeImage, 'camera/image', 10,
    intra_process_communication=True)

# No memory copying when publisher and subscriber in same process
```

## 📞 Request-Response Pattern

### Concept Overview

The request-response (service) pattern provides synchronous, one-to-one communication. A client sends a request and waits for a response from a service server.

### Service Implementation

```python
# Service Definition: example_interfaces/srv/AddTwoInts
# int64 a
# int64 b
# ---
# int64 sum

# Service Server
class MathService(Node):
    def __init__(self):
        super().__init__('math_service')

        self.service = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

        self.get_logger().info('Math Service started')

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b

        self.get_logger().info(
            f'Incoming request: {request.a} + {request.b} = {response.sum}')

        # Simulate processing time
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

        return response

# Service Client
class MathClient(Node):
    def __init__(self):
        super().__init__('math_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for math service...')

        self.get_logger().info('Math client started')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

# Complex Robotics Service: Path Planning
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped

class PathPlanningService(Node):
    def __init__(self):
        super().__init__('path_planning_service')

        self.service = self.create_service(
            GetPlan, 'plan_path', self.plan_path_callback)

        # Simulate map data
        self.obstacles = [
            (2.0, 2.0), (3.0, 1.0), (1.0, 3.0),  # Obstacle coordinates
            (4.0, 4.0), (5.0, 2.0)
        ]

    def plan_path_callback(self, request, response):
        """Plan path from start to goal avoiding obstacles"""
        start = request.start
        goal = request.goal

        self.get_logger().info(f'Planning path from ({start.pose.position.x}, {start.pose.position.y}) '
                              f'to ({goal.pose.position.x}, {goal.pose.position.y})')

        # Simple A* path planning algorithm
        path = self.plan_astar(start.pose.position, goal.pose.position)

        response.plan.poses = path
        response.plan.header.stamp = self.get_clock().now().to_msg()
        response.plan.header.frame_id = 'map'

        self.get_logger().info(f'Generated path with {len(path)} waypoints')

        return response

    def plan_astar(self, start, goal):
        """Simplified A* path planning"""
        # Simplified path planning - in reality would use proper A* algorithm
        path = []

        # Direct path with obstacle avoidance
        start_x, start_y = start.x, start.y
        goal_x, goal_y = goal.x, goal.y

        steps = 10
        for i in range(steps + 1):
            t = i / steps
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)

            # Add some deviation to avoid obstacles
            for obs_x, obs_y in self.obstacles:
                if abs(x - obs_x) < 0.5 and abs(y - obs_y) < 0.5:
                    y += 0.5  # Deviate around obstacle

            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path.append(pose)

        return path
```

### Asynchronous Service Calls

```python
import asyncio
from rclpy.executors import MultiThreadedExecutor

class AsyncMathClient(Node):
    def __init__(self):
        super().__init__('async_math_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

    async def send_request_async(self, a, b):
        """Asynchronous service call"""
        if not self.client.wait_for_service(timeout_sec=1.0):
            raise Exception('Service not available')

        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        future = self.client.call_async(request)

        # Wait for future to complete
        response = await future
        return response.sum

    async def batch_calculations(self):
        """Process multiple requests concurrently"""
        tasks = []

        # Create multiple concurrent requests
        for i in range(5):
            task = self.send_request_async(i, i*2)
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            self.get_logger().info(f'{i} + {i*2} = {result}')
```

## 🎯 Action Pattern

### Concept Overview

Actions are designed for long-running tasks that provide periodic feedback and can be canceled before completion. This pattern is ideal for robot navigation, manipulation tasks, and complex computations.

### Action Implementation

```python
# Action Definition: control_msgs/action/FollowJointTrajectory
# trajectory_msgs/JointTrajectory trajectory
# duration tolerance_time
# ---
# control_msgs/FollowJointTrajectoryResult result
# joint_names[]
# success bool
# ---
# control_msgs/FollowJointTrajectoryFeedback feedback
# joint_names[]
# actual.positions[]

# Action Server: Robot Arm Controller
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import threading

class RobotArmController(Node):
    def __init__(self):
        super().__init__('robot_arm_controller')

        self.action_server = rclpy_action.ActionServer(
            self,
            FollowJointTrajectory,
            'follow_joint_trajectory',
            self.execute_trajectory)

        # Simulate robot arm state
        self.joint_names = ['joint_1', 'joint_2', 'joint_3']
        self.current_positions = [0.0, 0.0, 0.0]

        self.get_logger().info('Robot Arm Action Server started')

    def execute_trajectory(self, goal_handle):
        """Execute robot arm trajectory with feedback"""
        self.get_logger().info('Executing trajectory...')

        trajectory = goal_handle.request.trajectory

        feedback_msg = FollowJointTrajectory.Feedback()
        feedback_msg.joint_names = self.joint_names

        try:
            for point in trajectory.points:
                # Simulate movement time
                duration = 0.1
                start_time = self.get_clock().now()

                while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration:
                    # Check if goal was canceled
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        return FollowJointTrajectory.Result()

                    # Simulate smooth movement
                    for i, target_pos in enumerate(point.positions):
                        current = self.current_positions[i]
                        target = target_pos

                        # Linear interpolation
                        self.current_positions[i] = current + (target - current) * 0.1

                    # Send feedback
                    feedback_msg.actual.positions = self.current_positions
                    goal_handle.publish_feedback(feedback_msg)

                    rclpy.spin_once(self, timeout_sec=0.01)

                # Update to target position
                self.current_positions = point.positions.copy()

            # Success
            goal_handle.succeed()
            result = FollowJointTrajectory.Result()
            result.success = True
            result.joint_names = self.joint_names

            self.get_logger().info('Trajectory executed successfully')
            return result

        except Exception as e:
            self.get_logger().error(f'Trajectory execution failed: {e}')
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.success = False
            return result

# Action Client: Robot Navigation
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Pose

class NavigationClient(Node):
    def __init__(self):
        super().__init__('navigation_client')

        self.action_client = rclpy_action.ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info('Navigation Action Client started')

    def send_goal(self, x, y, theta=0.0):
        """Send navigation goal"""
        goal_msg = NavigateToPose.Goal()

        # Set target pose
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Set orientation
        import math
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2)

        # Send goal with callback options
        send_goal_options = rclpy_action.client.ClientGoalWrapper(
            self.action_client,
            goal_msg,
            self.goal_response_callback,
            self.feedback_callback,
            self.result_callback
        )

        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        future.add_done_callback(self.goal_response_callback)

        return future

    def goal_response_callback(self, future):
        """Handle goal acceptance/rejection"""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback):
        """Handle navigation feedback"""
        current_pose = feedback.feedback.current_pose
        distance_remaining = feedback.feedback.distance_remaining

        self.get_logger().info(
            f'Distance remaining: {distance_remaining:.2f}m')

    def result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result

        if result:
            self.get_logger().info('Navigation completed successfully!')
        else:
            self.get_logger().info('Navigation failed')
```

### Action Usage Patterns

```python
# Using actions in applications
class RobotApplication(Node):
    def __init__(self):
        super().__init__('robot_application')

        # Create multiple action clients
        self.nav_client = NavigationClient()
        self.arm_client = RobotArmController()

        # Track task states
        self.current_task = None

    async def execute_pick_and_place(self, pick_location, place_location):
        """Coordinate navigation and manipulation"""

        # 1. Navigate to pick location
        self.get_logger().info('Navigating to pick location...')
        nav_goal = self.nav_client.send_goal(
            pick_location['x'], pick_location['y'])

        # Wait for navigation to complete
        await self.wait_for_action_result(nav_goal)

        # 2. Pick up object (arm motion)
        self.get_logger().info('Picking up object...')
        pick_trajectory = self.create_pickup_trajectory()
        arm_goal = self.arm_client.send_goal_async(pick_trajectory)

        # Wait for arm motion
        await self.wait_for_action_result(arm_goal)

        # 3. Navigate to place location
        self.get_logger().info('Navigating to place location...')
        nav_goal = self.nav_client.send_goal(
            place_location['x'], place_location['y'])

        await self.wait_for_action_result(nav_goal)

        # 4. Place object (arm motion)
        self.get_logger().info('Placing object...')
        place_trajectory = self.create_place_trajectory()
        arm_goal = self.arm_client.send_goal_async(place_trajectory)

        await self.wait_for_action_result(arm_goal)

        self.get_logger().info('Pick and place completed!')

    def create_pickup_trajectory(self):
        """Create pickup trajectory for robot arm"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3']

        # Approach position
        point1 = JointTrajectoryPoint()
        point1.positions = [0.5, -0.3, 0.2]
        point1.time_from_start.sec = 1

        # Grasp position
        point2 = JointTrajectoryPoint()
        point2.positions = [0.5, -0.3, 0.0]
        point2.time_from_start.sec = 2

        trajectory.points = [point1, point2]
        return trajectory
```

## 📊 Parameter Pattern

### Concept Overview

Parameters provide a mechanism for dynamic configuration of nodes without requiring restart. This pattern is essential for runtime adjustment of behavior, calibration values, and system settings.

### Parameter Implementation

```python
class ConfigurableRobot(Node):
    def __init__(self):
        super().__init__('configurable_robot')

        # Declare parameters with default values and descriptions
        self.declare_parameter('max_speed', 2.0)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('controller_frequency', 50.0)
        self.declare_parameter('emergency_stop_enabled', True)

        # Complex parameter structures
        self.declare_parameter(
            'sensor_config',
            {
                'camera_fps': 30,
                'laser_frequency': 10,
                'camera_resolution': [640, 480]
            }
        )

        # Get parameter values
        self.max_speed = self.get_parameter('max_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value

        # Set up parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Timer for using parameter values
        self.timer = self.create_timer(
            1.0 / self.get_parameter('controller_frequency').value,
            self.control_loop)

        self.get_logger().info('Configurable Robot started')

    def parameters_callback(self, params):
        """Handle parameter changes dynamically"""
        result = SetParametersResult()
        result.successful = True

        for param in params:
            if param.name == 'max_speed':
                if param.value > 0.0 and param.value <= 10.0:
                    self.max_speed = param.value
                    self.get_logger().info(f'Updated max_speed to {self.max_speed}')
                else:
                    result.successful = False
                    result.reason = 'max_speed must be between 0 and 10'

            elif param.name == 'safety_distance':
                if param.value > 0.1 and param.value <= 5.0:
                    self.safety_distance = param.value
                    self.get_logger().info(f'Updated safety_distance to {self.safety_distance}')
                else:
                    result.successful = False
                    result.reason = 'safety_distance must be between 0.1 and 5.0'

            elif param.name == 'controller_frequency':
                if param.value > 1.0 and param.value <= 200.0:
                    # Update timer frequency
                    self.timer.timer_period_ns = int(1e9 / param.value)
                    self.get_logger().info(f'Updated controller_frequency to {param.value}')
                else:
                    result.successful = False
                    result.reason = 'controller_frequency must be between 1 and 200'

        return result

    def control_loop(self):
        """Use parameter values in control logic"""
        # Use parameters in robot control
        current_speed = self.calculate_speed()

        if current_speed > self.max_speed:
            self.get_logger().warn(f'Speed {current_speed} exceeds max {self.max_speed}')
            self.apply_brakes()

        # Use safety distance
        obstacle_distance = self.get_obstacle_distance()
        if obstacle_distance < self.safety_distance:
            self.get_logger().warn(f'Obstacle too close: {obstacle_distance}m < {self.safety_distance}m')
            self.stop_movement()

# Parameter configuration from YAML file
# config/robot_params.yaml
"""
configurable_robot:
  ros__parameters:
    max_speed: 3.0
    safety_distance: 0.8
    controller_frequency: 100.0
    emergency_stop_enabled: true
    sensor_config:
      camera_fps: 60
      laser_frequency: 20
      camera_resolution: [1280, 720]
"""
```

### Parameter Management

```python
class ParameterManager(Node):
    def __init__(self):
        super().__init__('parameter_manager')

        # Load parameters from file
        self.load_parameters_from_file()

        # Set up parameter server for other nodes
        self.parameter_server = None

    def load_parameters_from_file(self):
        """Load parameters from YAML file"""
        import yaml
        import os

        config_file = 'config/robot_params.yaml'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Apply parameters to nodes
            for node_name, params in config.items():
                self.get_logger().info(f'Loading parameters for {node_name}')
                # In a real system, would send these to the appropriate nodes

# Command line parameter management
# ros2 param list
# ros2 param get /configurable_robot max_speed
# ros2 param set /configurable_robot max_speed 5.0
# ros2 param dump /configurable_robot robot_params.yaml
# ros2 param load /configurable_robot robot_params.yaml
```

## 🔄 Advanced Communication Patterns

### Composite Patterns

```python
class HybridRobotController(Node):
    def __init__(self):
        super().__init__('hybrid_robot_controller')

        # Mix of all communication patterns

        # Pub-Sub for continuous sensor data
        self.sensor_sub = self.create_subscription(
            LaserScan, 'laser_scan', self.sensor_callback, 10)

        self.control_pub = self.create_publisher(
            Twist, 'cmd_vel', 10)

        # Service for immediate commands
        self.emergency_service = self.create_service(
            Trigger, 'emergency_stop', self.emergency_stop_callback)

        # Action for navigation tasks
        self.nav_action_server = rclpy_action.ActionServer(
            self, NavigateToPose, 'navigate_to_pose', self.navigate_callback)

        # Parameters for configuration
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_radius', 0.5)

        # State management
        self.navigation_active = False
        self.emergency_stopped = False

    def sensor_callback(self, msg):
        """Process sensor data (pub-sub pattern)"""
        if self.emergency_stopped:
            return

        # Check for obstacles
        front_distance = min(msg.ranges[len(msg.ranges)//2-10:len(msg.ranges)//2+10])
        safety_radius = self.get_parameter('safety_radius').value

        if front_distance < safety_radius:
            self.stop_robot()
        elif self.navigation_active:
            # Continue navigation
            pass

    def emergency_stop_callback(self, request, response):
        """Handle immediate stop command (service pattern)"""
        self.emergency_stopped = True
        self.stop_robot()
        self.get_logger().warn('Emergency stop activated!')

        response.success = True
        response.message = 'Emergency stop activated'
        return response

    async def navigate_callback(self, goal_handle):
        """Handle navigation task (action pattern)"""
        self.navigation_active = True

        try:
            # Complex navigation logic with feedback
            target_pose = goal_handle.request.pose

            while not self.reached_target(target_pose):
                if goal_handle.is_cancel_requested():
                    self.navigation_active = False
                    goal_handle.canceled()
                    return NavigateToPose.Result()

                # Send feedback
                feedback = NavigateToPose.Feedback()
                feedback.current_pose = self.get_current_pose()
                feedback.distance_remaining = self.calculate_distance(target_pose)
                goal_handle.publish_feedback(feedback)

                # Control robot movement
                cmd = self.calculate_navigation_command(target_pose)
                self.control_pub.publish(cmd)

                await asyncio.sleep(0.1)

            # Success
            goal_handle.succeed()
            result = NavigateToPose.Result()
            result.success = True
            return result

        finally:
            self.navigation_active = False
```

### Communication Pattern Selection Guide

| Pattern | Best For | Latency | Throughput | Complexity |
|---------|----------|---------|------------|------------|
| **Pub-Sub** | Continuous data streams | Low | High | Low |
| **Service** | Immediate request/response | Medium | Low | Low |
| **Action** | Long-running tasks with feedback | High | Medium | High |
| **Parameter** | Configuration management | N/A | Low | Low |

### Pattern Decision Tree

```
Need Configuration?
├── Yes → Use Parameters
└── No → Is this a quick request/response?
    ├── Yes → Use Service
    └── No → Will this take time and need feedback?
        ├── Yes → Use Action
        └── No → Use Publish-Subscribe
```

## 🎯 Performance Optimization

### Communication Efficiency

```python
# Optimized message design
from std_msgs.msg import Header

# ❌ Bad: Large, complex messages
class BadSensorData:
    def __init__(self):
        self.timestamp = time.time()
        self.sensor_id = "sensor_001"
        self.data_type = "laser_scan"
        self.readings = [0.0] * 1000  # Fixed large array
        self.metadata = {
            "temperature": 25.0,
            "humidity": 50.0,
            "calibration": {...}
        }

# ✅ Good: Optimized, minimal messages
class GoodSensorData:
    def __init__(self):
        self.header = Header()  # Standard ROS header
        self.readings = []      # Only send actual readings
        self.max_range = 10.0   # Configurable range
```

### Memory Management

```python
# Reuse message objects
class EfficientPublisher(Node):
    def __init__(self):
        super().__init__('efficient_publisher')

        # Reuse message object to avoid memory allocation
        self.reusable_msg = Twist()
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.timer = self.create_timer(0.1, self.publish_command)

    def publish_command(self):
        # Reuse the same message object
        self.reusable_msg.linear.x = 1.0
        self.reusable_msg.angular.z = 0.5

        self.publisher.publish(self.reusable_msg)
```

---

## 🎉 Chapter Summary

Communication patterns form the language of robotic systems, enabling different components to exchange information effectively:

1. **Publish-Subscribe**: Best for continuous data streams and sensor data
2. **Request-Response**: Ideal for immediate queries and configuration commands
3. **Actions**: Perfect for long-running tasks with progress feedback
4. **Parameters**: Essential for dynamic configuration and runtime adjustments

The key to successful robotics architecture is choosing the right communication pattern for each interaction type and optimizing for performance, reliability, and maintainability.

**[← Back to Chapter 2: ROS 2 Architecture](02-ros2-architecture.md) | [Continue to Chapter 4: Distributed Systems →](04-distributed-systems.md)**

---

## 🧠 Knowledge Check

Test your understanding of Communication Patterns:

### Question 1
Which communication pattern is best for continuous sensor data streaming?

**Answer**
> **C. Publish-Subscribe**
>
> Publish-subscribe is ideal for continuous data streams from sensors because it provides one-to-many communication with decoupled producers and consumers.

---

### Question 2
What pattern should you use for a navigation task that takes minutes to complete?

**Answer**
> **B. Action**
>
> Actions are designed for long-running tasks that provide periodic feedback and can be canceled before completion.

---

### Question 3
Which pattern allows dynamic configuration without node restart?

**Answer**
> **A. Parameters**
>
> Parameters enable dynamic configuration changes at runtime without requiring node restarts.

---

### Question 4
What is the main advantage of the publish-subscribe pattern?

**Answer**
> **C. Decoupled one-to-many communication**
>
> The publish-subscribe pattern provides loose coupling between data producers and consumers, enabling flexible system architectures.