---
title: "Chapter 2: ROS 2 Architecture"
sidebar_label: "2. ROS 2 Architecture"
sidebar_position: 2
---

# Chapter 2: ROS 2 Architecture

## The Foundation of Modern Robotics Systems

ROS 2 (Robot Operating System 2) is not an operating system but a **middleware framework** that provides the essential tools, libraries, and conventions for building robotic applications. It serves as the nervous system that connects sensors, processors, and actuators in a coordinated, distributed architecture.

## 🧠 Core Architecture Concepts

### Design Philosophy

ROS 2 is built on several fundamental principles:

1. **Distributed Computing**: Nodes can run across multiple machines
2. **Real-time Capabilities**: Deterministic timing for critical operations
3. **Production-Ready**: Industrial-grade reliability and security
4. **Cross-Platform**: Linux, Windows, macOS support
5. **Multi-Language**: C++, Python, Java, and more

### Architectural Layers

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│  (Your robot's specific behaviors and algorithms)       │
├─────────────────────────────────────────────────────────┤
│                 Client Libraries                        │
│  (rclcpp, rclpy, rcljava - ROS 2 client libraries)     │
├─────────────────────────────────────────────────────────┤
│               ROS 2 Core (RCL)                          │
│     (Abstract ROS client library implementation)        │
├─────────────────────────────────────────────────────────┤
│                  DDS/RTPS Layer                         │
│     (Data Distribution Service - Real-Time Publish)     │
│         (e.g., Cyclone DDS, Fast DDS)                  │
├─────────────────────────────────────────────────────────┤
│                Transport Layer                          │
│           (UDP, TCP, Shared Memory)                     │
└─────────────────────────────────────────────────────────┘
```

## 🏗️ Key Components

### 1. Nodes

Nodes are the fundamental computational units in ROS 2. Each node is a single process that performs computation and communicates with other nodes.

```python
# Example: Simple ROS 2 node in Python
import rclpy
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info('Robot Controller Node started')

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Key Node Characteristics:**
- **Independent Processes**: Each node runs as a separate process
- **Named Identity**: Unique names for system identification
- **Communication**: Use topics, services, and actions to exchange data
- **Parameters**: Dynamic configuration without restart
- **Lifecycle**: Managed startup, activation, deactivation, and shutdown

### 2. Topics

Topics are named communication channels for one-to-many data exchange.

```cpp
// Example: Publisher in C++
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

class LaserScanner : public rclcpp::Node {
public:
    LaserScanner() : Node("laser_scanner") {
        publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
            "laser_data", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&LaserScanner::publish_scan, this));
    }

private:
    void publish_scan() {
        auto scan = std::make_unique<sensor_msgs::msg::LaserScan>();
        // Fill scan data...
        publisher_->publish(std::move(scan));
    }

    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};
```

**Topic Properties:**
- **Type Safety**: Strongly typed message interfaces
- **QoS Settings**: Quality of Service parameters (reliability, durability)
- **One-to-Many**: Single publisher to multiple subscribers
- **Asynchronous**: Decoupled communication pattern

### 3. Services

Services provide request-response communication for synchronous operations.

```python
# Example: Service Server
from example_interfaces.srv import AddTwoInts

class AddService(Node):
    def __init__(self):
        super().__init__('add_service')
        self.service = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Adding {request.a} + {request.b} = {response.sum}')
        return response
```

**Service Characteristics:**
- **Synchronous**: Request must wait for response
- **One-to-One**: Single client to single server
- **Blocking**: Client waits for operation completion
- **Stateless**: Each request is independent

### 4. Actions

Actions are for long-running tasks that provide feedback and can be canceled.

```cpp
// Example: Action Client for robot navigation
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

class NavigationClient : public rclcpp::Node {
public:
    NavigationClient() : Node("navigation_client") {
        action_client_ = rclcpp_action::create_client<NavigateToPose>(
            this, "navigate_to_pose");
    }

    void navigate_to_goal(const geometry_msgs::msg::PoseStamped& goal_pose) {
        auto goal_msg = NavigateToPose::Goal();
        goal_msg.pose = goal_pose;

        auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
        send_goal_options.feedback_callback =
            std::bind(&NavigationClient::feedback_callback, this, _1, _2);

        action_client_->async_send_goal(goal_msg, send_goal_options);
    }

private:
    void feedback_callback(
        GoalHandleNavigateToPose::SharedPtr,
        const std::shared_ptr<const NavigateToPose::Feedback> feedback) {
        RCLCPP_INFO(get_logger(), "Distance remaining: %.2f",
                   feedback->distance_remaining);
    }

    rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_;
};
```

**Action Properties:**
- **Long-Running**: Tasks that take significant time
- **Feedback**: Periodic status updates
- **Cancellable**: Can be aborted before completion
- **Goal-Oriented**: Clear start and end conditions

## 🔧 Quality of Service (QoS)

ROS 2's Quality of Service policies provide fine-grained control over communication behavior.

### QoS Policies

| Policy | Description | Common Settings |
|--------|-------------|-----------------|
| **Reliability** | Message delivery guarantees | RELIABLE, BEST_EFFORT |
| **Durability** | Data persistence for late joiners | VOLATILE, TRANSIENT_LOCAL |
| **Deadline** | Maximum time between messages | Custom duration |
| **Lifespan** | Message expiration time | Custom duration |
| **Liveliness** | Publisher health monitoring | AUTOMATIC, MANUAL_BY_TOPIC |
| **Lease Duration** | Liveliness timeout | Custom duration |

### QoS Profiles

```python
# Built-in QoS profiles
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# Sensor data profile (best effort, volatile)
sensor_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10
)

# State data profile (reliable, transient local)
state_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1
)

# Custom profile for real-time control
realtime_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    deadline=Duration(seconds=0, nanoseconds=10000000)  # 10ms
)
```

## 🏛️ DDS (Data Distribution Service)

ROS 2 uses DDS as its underlying communication middleware, providing:

### DDS Implementations

1. **Cyclone DDS** (Default)
   - Apache 2.0 licensed
   - High performance
   - Low memory footprint

2. **Fast DDS**
   - Advanced features
   - Real-time capabilities
   - Commercial support

3. **Connext DDS**
   - Enterprise-grade
   - Advanced security
   - Commercial licensing

### DDS Benefits

- **Decentralized Architecture**: No central broker
- **Discovery**: Automatic peer detection
- **Scalability**: Hundreds to thousands of nodes
- **Real-time**: Deterministic communication
- **Security**: Authentication, encryption, access control

## 🔐 Security Architecture

ROS 2 security is built on DDS security standards:

### Security Features

1. **Authentication**: Node identity verification
2. **Access Control**: Permission-based communication
3. **Encryption**: Data confidentiality
4. **Integrity**: Data tamper protection

### Security Implementation

```bash
# Create security artifacts
ros2 security generate_artifacts \
  --key-store ~/robot_security \
  --permissions-file robot_permissions.xml \
  --policy-file robot_security_policy.xml \
  --config-file robot_security_config.xml

# Run with security
export ROS_SECURITY_ROOT_DIR=~/robot_security
export ROS_SECURITY_ENABLE=true
ros2 run my_package my_node
```

## 📦 Package Architecture

ROS 2 packages are the unit of code organization:

### Package Structure

```
my_robot_package/
├── package.xml          # Package metadata and dependencies
├── CMakeLists.txt       # Build configuration (C++)
├── setup.py            # Build configuration (Python)
├── src/                # Source code (C++)
│   └── my_robot_node.cpp
├── scripts/            # Executable scripts (Python)
│   └── my_robot_script.py
├── launch/             # Launch files
│   └── robot.launch.py
├── config/             # Configuration files
│   └── robot_params.yaml
├── srv/                # Service definitions
├── msg/                # Message definitions
├── action/             # Action definitions
└── test/               # Test files
```

### Package Creation

```bash
# Create a new package
ros2 pkg create my_robot_package \
  --build-type ament_cmake \
  --dependencies rclcpp std_msgs

# Create Python package
ros2 pkg create my_robot_python \
  --build-type ament_python \
  --dependencies rclpy std_msgs
```

## 🚀 Launch System

The ROS 2 launch system orchestrates multiple nodes:

### Launch File (Python)

```python
# launch/robot_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='robot_1',
        description='Robot identifier'
    )

    # Define nodes
    robot_controller = Node(
        package='robot_control',
        executable='robot_controller',
        name=LaunchConfiguration('robot_name') + '_controller',
        parameters=[{'control_frequency': 50.0}],
        output='screen'
    )

    sensor_node = Node(
        package='robot_sensors',
        executable='sensor_processor',
        name=LaunchConfiguration('robot_name') + '_sensors',
        remappings=[('laser_scan', 'front_laser')],
        output='screen'
    )

    return LaunchDescription([
        robot_name_arg,
        robot_controller,
        sensor_node
    ])
```

## 🧪 Testing Architecture

ROS 2 provides comprehensive testing frameworks:

### Test Types

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Multi-node interaction testing
3. **System Tests**: Complete robot system testing
4. **Performance Tests**: Real-time and load testing

### Testing Framework

```python
# Test file: test_robot_controller.py
import unittest
import rclpy
from robot_control.robot_controller import RobotController

class TestRobotController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.robot = RobotController()

    def tearDown(self):
        self.robot.destroy_node()

    def test_initialization(self):
        self.assertIsNotNone(self.robot)
        self.assertEqual(self.robot.get_name(), 'robot_controller')

    def test_parameter_setting(self):
        param_value = 10.0
        self.robot.set_parameters([rclpy.Parameter('max_speed', value=param_value)])
        self.assertEqual(self.robot.get_parameter('max_speed').value, param_value)
```

## 📊 Performance Considerations

### Real-Time Performance

```cpp
// Real-time node configuration
class RealtimeNode : public rclcpp::Node {
public:
    RealtimeNode() : Node("realtime_node") {
        // Set real-time scheduler
        struct sched_param param;
        param.sched_priority = 90;
        sched_setscheduler(0, SCHED_FIFO, &param);

        // Configure QoS for real-time performance
        rclcpp::QoS realtime_qos(10);
        realtime_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
        realtime_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        realtime_qos.deadline(std::chrono::milliseconds(10));

        publisher_ = this->create_publisher<std_msgs::msg::Float64>(
            "control_output", realtime_qos);
    }
};
```

### Memory Management

```python
# Efficient message handling
class EfficientSubscriber(Node):
    def __init__(self):
        super().__init__('efficient_subscriber')

        # Use intra-process communication for zero-copy
        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            10,
            raw=True  # Raw message for better performance
        )

    def image_callback(self, msg):
        # Process message efficiently without copying
        # Access raw data directly from message
        pass
```

## 🔍 Debugging and Monitoring

### Visualization Tools

1. **RViz 2**: 3D visualization
2. **Plot Juggler**: Data plotting and analysis
3. **Foxglove Studio**: Web-based visualization
4. **rqt**: Qt-based debugging tools

### Command Line Tools

```bash
# Node monitoring
ros2 node list                          # List all nodes
ros2 node info /robot_controller        # Node details

# Topic inspection
ros2 topic list                         # List all topics
ros2 topic echo /laser_scan            # Display topic data
ros2 topic hz /laser_scan              # Topic frequency

# Service calls
ros2 service list                       # List all services
ros2 service call /add_two_ints         # Call a service

# Parameter management
ros2 param list                         # List all parameters
ros2 param get /robot_controller max_speed
ros2 param set /robot_controller max_speed 5.0
```

## 🎯 Best Practices

### Architecture Guidelines

1. **Modularity**: Single responsibility per node
2. **Decoupling**: Minimize direct node dependencies
3. **Configuration**: Use parameters for runtime settings
4. **Testing**: Comprehensive test coverage
5. **Documentation**: Clear interfaces and usage examples

### Performance Guidelines

1. **QoS Optimization**: Match QoS to application needs
2. **Message Design**: Efficient message structures
3. **Resource Management**: Monitor CPU and memory usage
4. **Real-time Constraints**: Consider timing requirements
5. **Profiling**: Use performance analysis tools

## 📚 Further Learning

### Documentation Resources

- [ROS 2 Official Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Design Documentation](https://design.ros2.org/)
- [DDS Standards](https://www.omg.org/spec/DDS/)

### Practice Exercises

1. **Multi-Node System**: Create a robot with separate sensor, control, and actuator nodes
2. **QoS Experimentation**: Test different QoS settings for various scenarios
3. **Launch Configuration**: Design complex launch systems with parameters and remappings
4. **Security Implementation**: Set up secure communication between nodes
5. **Performance Optimization**: Profile and optimize node performance

---

## 🎉 Chapter Summary

ROS 2 architecture provides a robust, scalable foundation for modern robotics applications. Key takeaways:

1. **Distributed Design**: Nodes communicate independently across multiple machines
2. **Communication Patterns**: Topics, services, and actions cover all interaction needs
3. **Quality of Service**: Fine-grained control over communication behavior
4. **Security**: Enterprise-grade security features built on DDS standards
5. **Real-time Capabilities**: Support for deterministic, time-critical applications

**[← Back to Quarter 1 Overview](index.md) | [Continue to Chapter 3: Communication Patterns →](03-communication-patterns.md)**

---

## 🧠 Knowledge Check

Test your understanding of ROS 2 Architecture concepts:

### Question 1
What is the role of DDS in ROS 2 architecture?

**Answer**
> **C. Handles underlying communication and data distribution**
>
> DDS (Data Distribution Service) provides the real-time publish-subscribe middleware that enables ROS 2's distributed communication capabilities.

---

### Question 2
Which QoS policy controls message delivery guarantees?

**Answer**
> **C. Reliability**
>
> The Reliability QoS policy determines whether messages must be delivered reliably (RELIABLE) or can be dropped (BEST_EFFORT).

---

### Question 3
What communication pattern is best for long-running tasks with feedback?

**Answer**
> **C. Actions**
>
> Actions are designed for long-running operations that provide periodic feedback and can be canceled before completion.