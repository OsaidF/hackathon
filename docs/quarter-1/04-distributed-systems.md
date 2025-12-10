---
title: "Chapter 4: Distributed Systems"
sidebar_label: "4. Distributed Systems"
sidebar_position: 4
---

# Chapter 4: Distributed Systems

## Scaling Robotics Across Multiple Computers

Modern robots are increasingly complex systems that often require multiple computers working together seamlessly. A single computer may not have sufficient processing power, I/O capabilities, or physical placement to handle all robot functions. Distributed systems enable us to scale computational resources, improve reliability, and optimize physical organization of robotic components.

## 🏗️ Distributed Architecture Concepts

### Why Distributed Robotics?

1. **Computational Scaling**: Multiple processors for AI, vision, control
2. **Physical Constraints**: Sensors and actuators in different locations
3. **Specialized Hardware**: GPUs for AI, FPGAs for real-time control
4. **Reliability**: Redundancy and fault tolerance
5. **Modularity**: Independent development and deployment

### Architecture Patterns

```
┌─────────────────────────────────────────────────────────────┐
│                    Robot Network Topology                  │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Control   │    │ Perception  │    │   Planning  │     │
│  │   Computer  │◄──►│   Computer  │◄──►│   Computer  │     │
│  │             │    │             │    │             │     │
│  │ Motor Ctrl  │    │   Vision    │    │ Navigation │     │
│  │ Safety Sys  │    │   Sensors   │    │   AI ML    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             │                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Network Backbone                        │   │
│  │         (Ethernet, WiFi, 5G, CAN bus)                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🌐 Network Infrastructure

### Network Types and Characteristics

| Network Type | Bandwidth | Latency | Range | Reliability | Use Cases |
|--------------|-----------|---------|-------|-------------|-----------|
| **Ethernet** | 1–10 Gbps | &lt;1ms | 100m | Very High | Real-time control, high-bandwidth sensors |
| **WiFi 6** | 1–9 Gbps | 5–20ms | 100m | Medium | Mobile robots, wireless sensors |
| **5G** | 100–1000 Mbps | 1–10ms | km | High | Outdoor mobile robots, cloud robotics |
| **CAN Bus** | 1 Mbps | &lt;1ms | 1km | Very High | Embedded control, safety-critical systems |

### Network Configuration

```python
# Network discovery and configuration
class NetworkManager(Node):
    def __init__(self):
        super().__init__('network_manager')

        # Network configuration parameters
        self.declare_parameter('network_interface', 'eth0')
        self.declare_parameter('dds_domain', 42)
        self.declare_parameter('multicast_address', '239.255.0.1')

        # Network monitoring
        self.network_stats_timer = self.create_timer(
            1.0, self.monitor_network)

        # Node discovery service
        self.discovery_service = self.create_service(
            GetNetworkInfo, 'get_network_info', self.discovery_callback)

        self.known_nodes = {}
        self.network_health = {}

    def monitor_network(self):
        """Monitor network health and node connectivity"""
        # Check network interface status
        interface = self.get_parameter('network_interface').value

        # Get network statistics (would use system calls)
        stats = self.get_interface_stats(interface)

        # Monitor known nodes
        for node_name, node_info in self.known_nodes.items():
            if self.ping_node(node_info['address']):
                if node_name not in self.network_health:
                    self.get_logger().info(f'Node {node_name} is online')
                self.network_health[node_name] = {
                    'status': 'online',
                    'last_seen': self.get_clock().now()
                }
            else:
                if node_name in self.network_health:
                    self.get_logger().warn(f'Node {node_name} is offline')
                    self.network_health[node_name] = {
                        'status': 'offline',
                        'last_seen': self.network_health[node_name]['last_seen']
                    }

    def get_interface_stats(self, interface):
        """Get network interface statistics"""
        # In reality, would use psutil or similar library
        return {
            'bytes_sent': 1000000,
            'bytes_received': 2000000,
            'packets_dropped': 0,
            'errors': 0
        }

    def ping_node(self, address):
        """Check if a node is reachable"""
        # Simplified ping check
        import subprocess
        try:
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '1', address],
                capture_output=True, text=True, timeout=2
            )
            return result.returncode == 0
        except:
            return False

    def discovery_callback(self, request, response):
        """Provide network and node discovery information"""
        response.network_interface = self.get_parameter('network_interface').value
        response.dds_domain = self.get_parameter('dds_domain').value

        # Add known nodes to response
        for node_name, node_info in self.known_nodes.items():
            node_status = NetworkNode()
            node_status.name = node_name
            node_status.address = node_info['address']
            node_status.status = self.network_health.get(node_name, {}).get('status', 'unknown')
            response.nodes.append(node_status)

        return response
```

### DDS Domain Configuration

```bash
# Environment variables for DDS domain configuration
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterface>eth0</NetworkInterface></General></Domain></CycloneDDS>'

# CycloneDDS configuration file (cyclonedds.xml)
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain>
        <General>
            <NetworkInterfaceAddress>auto</NetworkInterfaceAddress>
            <AllowMulticast>true</AllowMulticast>
            <MulticastReceiveAddress>239.255.0.1</MulticastReceiveAddress>
            <MaxMessageSize>65536</MaxMessageSize>
            <FragmentSize>8192</FragmentSize>
        </General>

        <Internal>
            <FragmentSize>8192</FragmentSize>
            <TransportPriority>100</TransportPriority>
        </Internal>

        <Discovery>
            <MulticastInitialPeers>
                <Peer address="239.255.0.1"/>
            </MulticastInitialPeers>
            <ParticipantIndex>auto</ParticipantIndex>
            <MaxAutoParticipantIndex>99</MaxAutoParticipantIndex>
        </Discovery>

        <Tracing>
            <Verbosity>info</Verbosity>
            <OutputFile>cyclonedds.log</OutputFile>
        </Tracing>
    </Domain>
</CycloneDDS>
```

## 🤖 Multi-Robot Coordination

### Robot Team Architecture

```python
# Multi-robot coordination system
class MultiRobotCoordinator(Node):
    def __init__(self, robot_id):
        super().__init__(f'coordinator_{robot_id}')

        self.robot_id = robot_id
        self.team_members = []
        self.current_mission = None

        # Communication with team
        self.team_publisher = self.create_publisher(
            RobotStatus, 'team_status', 10)

        self.team_subscriber = self.create_subscription(
            RobotStatus, 'team_status', self.team_status_callback, 10)

        # Mission coordination
        self.mission_subscriber = self.create_subscription(
            MissionPlan, 'mission_plan', self.mission_callback, 10)

        # Task assignment
        self.task_client = self.create_client(
            AssignTask, 'assign_task')

        # Formation control
        self.formation_publisher = self.create_publisher(
            FormationGoal, 'formation_goal', 10)

        # Collision avoidance
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, f'robot_{robot_id}/pose', 10)

        self.get_logger().info(f'Multi-robot coordinator {robot_id} initialized')

    def team_status_callback(self, msg):
        """Process status updates from team members"""
        if msg.robot_id != self.robot_id:
            # Update team member information
            member_info = {
                'robot_id': msg.robot_id,
                'position': (msg.position.x, msg.position.y),
                'battery': msg.battery_level,
                'status': msg.status,
                'capabilities': msg.capabilities,
                'timestamp': msg.header.stamp
            }

            self.team_members.append(member_info)

            # Keep only recent updates
            self.team_members = [
                m for m in self.team_members
                if (self.get_clock().now() - m['timestamp']).nanoseconds / 1e9 < 10.0
            ]

            self.get_logger().info(f'Received status from robot {msg.robot_id}')

    def mission_callback(self, msg):
        """Handle new mission assignment"""
        self.current_mission = msg
        self.get_logger().info(f'Received mission: {msg.mission_type}')

        # Start mission execution
        self.execute_mission()

    def execute_mission(self):
        """Execute assigned mission with team coordination"""
        if self.current_mission.mission_type == 'area_coverage':
            self.coordinate_area_coverage()
        elif self.current_mission.mission_type == 'formation_patrol':
            self.coordinate_formation_patrol()
        elif self.current_mission.mission_type == 'collaborative_search':
            self.coordinate_search()

    def coordinate_area_coverage(self):
        """Coordinate team for area coverage"""
        mission_area = self.current_mission.area
        team_size = len(self.team_members) + 1  # +1 for self

        # Calculate individual coverage zones
        zones = self.divide_area(mission_area, team_size)

        # Assign zones based on robot capabilities and positions
        assignments = self.assign_zones(zones)

        # Execute assigned zone
        my_zone = assignments.get(self.robot_id)
        if my_zone:
            self.navigate_to_zone(my_zone)

    def divide_area(self, area, num_robots):
        """Divide mission area into zones for each robot"""
        # Simple grid-based division
        min_x, max_x = area.min_x, area.max_x
        min_y, max_y = area.min_y, area.max_y

        # Calculate grid dimensions
        cols = int(math.sqrt(num_robots))
        rows = math.ceil(num_robots / cols)

        zone_width = (max_x - min_x) / cols
        zone_height = (max_y - min_y) / rows

        zones = []
        for i in range(num_robots):
            row = i // cols
            col = i % cols

            zone = {
                'min_x': min_x + col * zone_width,
                'max_x': min_x + (col + 1) * zone_width,
                'min_y': min_y + row * zone_height,
                'max_y': min_y + (row + 1) * zone_height,
                'zone_id': i
            }
            zones.append(zone)

        return zones

    def assign_zones(self, zones):
        """Assign zones to robots based on capabilities and positions"""
        assignments = {}

        # Include self
        my_position = self.get_current_position()
        all_robots = [{'robot_id': self.robot_id, 'position': my_position}]

        # Add team members
        for member in self.team_members:
            all_robots.append(member)

        # Assign zones using simple distance-based algorithm
        unassigned_zones = zones.copy()

        for robot in sorted(all_robots,
                          key=lambda r: self.calculate_robot_priority(r)):
            if unassigned_zones:
                # Find closest zone
                best_zone = min(unassigned_zones,
                              key=lambda z: self.distance_to_zone(robot['position'], z))
                assignments[robot['robot_id']] = best_zone
                unassigned_zones.remove(best_zone)

        return assignments

    def coordinate_formation_patrol(self):
        """Coordinate team formation for patrol mission"""
        formation = self.current_mission.formation

        # Calculate formation positions
        formation_positions = self.calculate_formation(formation)

        # Assign positions to team members
        my_position = formation_positions.get(self.robot_id)

        if my_position:
            formation_goal = FormationGoal()
            formation_goal.formation_type = formation.type
            formation_goal.target_position.x = my_position[0]
            formation_goal.target_position.y = my_position[1]
            formation_goal.target_position.theta = my_position[2]

            self.formation_publisher.publish(formation_goal)

    def publish_status(self):
        """Publish current robot status to team"""
        status = RobotStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.robot_id = self.robot_id

        current_pose = self.get_current_pose()
        status.position.x = current_pose.position.x
        status.position.y = current_pose.position.y
        status.battery_level = self.get_battery_level()
        status.status = 'active'

        self.team_publisher.publish(status)

    def navigate_to_zone(self, zone):
        """Navigate to assigned coverage zone"""
        # Calculate zone center
        center_x = (zone['min_x'] + zone['max_x']) / 2
        center_y = (zone['min_y'] + zone['max_y']) / 2

        # Send navigation goal
        self.send_navigation_goal(center_x, center_y)

        self.get_logger().info(f'Navigating to zone {zone["zone_id"]}')
```

### Collaborative Perception

```python
# Multi-robot sensor fusion and perception
class CollaborativePerception(Node):
    def __init__(self):
        super().__init__('collaborative_perception')

        # Local sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, 'laser_scan', self.laser_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, 'camera/image', self.camera_callback, 10)

        # Shared perception data
        self.shared_map_publisher = self.create_publisher(
            OccupancyGrid, 'shared_map', 10)

        self.shared_detection_publisher = self.create_publisher(
            DetectionArray, 'shared_detections', 10)

        # Receive data from team
        self.shared_map_subscriber = self.create_subscription(
            OccupancyGrid, 'shared_map', self.shared_map_callback, 10)

        self.shared_detection_subscriber = self.create_subscription(
            DetectionArray, 'shared_detections', self.shared_detection_callback, 10)

        # Fusion state
        self.global_map = None
        self.team_detections = []
        self.last_publish_time = self.get_clock().now()

        self.get_logger().info('Collaborative perception system started')

    def laser_callback(self, msg):
        """Process local laser scan data"""
        # Convert laser scan to occupancy grid
        local_map = self.laser_to_occupancy_grid(msg)

        # Fuse with global map
        if self.global_map is None:
            self.global_map = local_map
        else:
            self.global_map = self.fuse_maps(self.global_map, local_map)

        # Publish updates periodically
        current_time = self.get_clock().now()
        if (current_time - self.last_publish_time).nanoseconds / 1e9 > 1.0:
            self.publish_shared_map()
            self.last_publish_time = current_time

    def camera_callback(self, msg):
        """Process camera data and detect objects"""
        # Perform object detection (simplified)
        detections = self.detect_objects(msg)

        # Publish detections
        detection_array = DetectionArray()
        detection_array.header = msg.header
        detection_array.detections = detections

        self.shared_detection_publisher.publish(detection_array)

    def detect_objects(self, image_msg):
        """Simplified object detection"""
        # In reality, would use neural network or computer vision algorithms
        detections = []

        # Simulate detecting some objects
        for i in range(2):
            detection = Detection()
            detection.class_id = f'object_{i}'
            detection.confidence = 0.8 + i * 0.1
            detection.bbox.x = 100 + i * 200
            detection.bbox.y = 100 + i * 100
            detection.bbox.width = 50
            detection.bbox.height = 50
            detection.robot_id = 'self'

            detections.append(detection)

        return detections

    def shared_map_callback(self, msg):
        """Receive shared map updates from team members"""
        # Fuse received map with global map
        if self.global_map is None:
            self.global_map = msg
        else:
            self.global_map = self.fuse_maps(self.global_map, msg)

    def shared_detection_callback(self, msg):
        """Receive detection data from team members"""
        # Add team detections to shared list
        self.team_detections.extend(msg.detections)

        # Keep only recent detections
        current_time = self.get_clock().now()
        self.team_detections = [
            d for d in self.team_detections
            if (current_time - self.get_clock().from_msg(d.header.stamp)).nanoseconds / 1e9 < 5.0
        ]

    def fuse_maps(self, map1, map2):
        """Fuse two occupancy grid maps"""
        # Simplified map fusion - in reality would use proper sensor fusion
        fused_map = OccupancyGrid()
        fused_map.header = self.get_clock().now().to_msg()
        fused_map.info = map1.info  # Assume same grid parameters

        # Combine occupancy probabilities
        fused_data = []
        for i in range(len(map1.data)):
            if i < len(map2.data):
                # Combine using probability theory
                prob1 = map1.data[i] / 100.0
                prob2 = map2.data[i] / 100.0

                # Combine odds ratios
                odds1 = prob1 / (1.0 - prob1 + 0.001)
                odds2 = prob2 / (1.0 - prob2 + 0.001)
                combined_odds = odds1 * odds2
                combined_prob = combined_odds / (1.0 + combined_odds)

                fused_data.append(int(combined_prob * 100))
            else:
                fused_data.append(map1.data[i])

        fused_map.data = fused_data
        return fused_map

    def publish_shared_map(self):
        """Publish the fused global map"""
        if self.global_map:
            self.global_map.header.stamp = self.get_clock().now().to_msg()
            self.shared_map_publisher.publish(self.global_map)
```

## 🔧 System Integration and Debugging

### Distributed System Health Monitoring

```python
class SystemHealthMonitor(Node):
    def __init__(self):
        super().__init__('system_health_monitor')

        # Health monitoring parameters
        self.declare_parameter('monitor_interval', 1.0)
        self.declare_parameter('timeout_threshold', 5.0)
        self.declare_parameter('max_cpu_usage', 80.0)
        self.declare_parameter('max_memory_usage', 85.0)

        # System metrics collection
        self.system_metrics = {}
        self.node_health = {}

        # Health data publishers
        self.health_publisher = self.create_publisher(
            SystemHealth, 'system_health', 10)

        self.alert_publisher = self.create_publisher(
            SystemAlert, 'system_alert', 10)

        # Monitoring timer
        self.monitor_timer = self.create_timer(
            self.get_parameter('monitor_interval').value,
            self.monitor_system)

        # Node discovery
        self.node_discovery = NodeDiscovery(self)

        self.get_logger().info('System Health Monitor started')

    def monitor_system(self):
        """Monitor overall system health"""
        # Collect system metrics
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()
        network_stats = self.get_network_stats()
        disk_usage = self.get_disk_usage()

        # Update system metrics
        self.system_metrics = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'network_stats': network_stats,
            'disk_usage': disk_usage,
            'timestamp': self.get_clock().now()
        }

        # Monitor individual nodes
        self.monitor_nodes()

        # Check for alerts
        self.check_alerts()

        # Publish health status
        self.publish_health()

    def monitor_nodes(self):
        """Monitor individual node health"""
        # Update node discovery
        self.node_discovery.discover_nodes()

        # Check each known node
        for node_name, node_info in self.node_discovery.get_nodes().items():
            health_status = self.check_node_health(node_name, node_info)
            self.node_health[node_name] = health_status

    def check_node_health(self, node_name, node_info):
        """Check health of individual node"""
        health = NodeHealth()
        health.node_name = node_name
        health.status = 'healthy'
        health.cpu_usage = 0.0
        health.memory_usage = 0.0
        health.topic_count = 0
        health.service_count = 0

        # Check if node is responsive
        if self.is_node_responsive(node_info):
            # Get node statistics (simplified)
            try:
                stats = self.get_node_statistics(node_name)
                health.cpu_usage = stats.get('cpu', 0.0)
                health.memory_usage = stats.get('memory', 0.0)
                health.topic_count = stats.get('topics', 0)
                health.service_count = stats.get('services', 0)

                # Check thresholds
                if health.cpu_usage > self.get_parameter('max_cpu_usage').value:
                    health.status = 'warning'
                if health.memory_usage > self.get_parameter('max_memory_usage').value:
                    health.status = 'critical'

            except Exception as e:
                self.get_logger().warn(f'Failed to get stats for {node_name}: {e}')
                health.status = 'warning'
        else:
            health.status = 'offline'

        return health

    def is_node_responsive(self, node_info):
        """Check if node is responsive"""
        # Simplified check - in reality would ping node or check last activity
        return True

    def get_node_statistics(self, node_name):
        """Get detailed node statistics"""
        # In reality, would use ROS 2 node statistics API
        return {
            'cpu': 25.0 + random.uniform(-10, 10),
            'memory': 40.0 + random.uniform(-10, 20),
            'topics': 5,
            'services': 2
        }

    def check_alerts(self):
        """Check for system alerts"""
        alerts = []

        # CPU usage alert
        if self.system_metrics['cpu_usage'] > 90.0:
            alert = SystemAlert()
            alert.level = 'critical'
            alert.message = f'High CPU usage: {self.system_metrics["cpu_usage"]:.1f}%'
            alert.component = 'system'
            alerts.append(alert)

        # Memory usage alert
        if self.system_metrics['memory_usage'] > 90.0:
            alert = SystemAlert()
            alert.level = 'critical'
            alert.message = f'High memory usage: {self.system_metrics["memory_usage"]:.1f}%'
            alert.component = 'system'
            alerts.append(alert)

        # Node offline alerts
        for node_name, health in self.node_health.items():
            if health.status == 'offline':
                alert = SystemAlert()
                alert.level = 'warning'
                alert.message = f'Node {node_name} is offline'
                alert.component = 'node'
                alerts.append(alert)

        # Publish alerts
        for alert in alerts:
            self.alert_publisher.publish(alert)

    def publish_health(self):
        """Publish system health status"""
        health_msg = SystemHealth()
        health_msg.header.stamp = self.get_clock().now().to_msg()

        health_msg.cpu_usage = self.system_metrics['cpu_usage']
        health_msg.memory_usage = self.system_metrics['memory_usage']
        health_msg.disk_usage = self.system_metrics['disk_usage']
        health_msg.active_nodes = len(self.node_health)
        health_msg.healthy_nodes = sum(1 for h in self.node_health.values() if h.status == 'healthy')

        # Add node health details
        for node_name, node_health in self.node_health.items():
            health_msg.node_health.append(node_health)

        self.health_publisher.publish(health_msg)

# Node discovery utility
class NodeDiscovery:
    def __init__(self, parent_node):
        self.parent = parent_node
        self.known_nodes = {}
        self.last_discovery = self.parent.get_clock().now()

    def discover_nodes(self):
        """Discover ROS 2 nodes in the system"""
        current_time = self.parent.get_clock().now()

        # Run discovery periodically
        if (current_time - self.last_discovery).nanoseconds / 1e9 > 5.0:
            # Use ROS 2 node discovery (simplified)
            # In reality would use proper ROS 2 discovery API
            self.discover_via_cli()
            self.last_discovery = current_time

    def discover_via_cli(self):
        """Discover nodes using CLI"""
        try:
            # Use ros2 node list command
            result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                nodes = result.stdout.strip().split('\n')
                for node in nodes:
                    node = node.strip()
                    if node and node not in self.known_nodes:
                        self.known_nodes[node] = {
                            'discovered_time': self.parent.get_clock().now(),
                            'last_seen': self.parent.get_clock().now()
                        }
                    elif node in self.known_nodes:
                        self.known_nodes[node]['last_seen'] = self.parent.get_clock().now()

        except Exception as e:
            self.parent.get_logger().warn(f'Node discovery failed: {e}')

    def get_nodes(self):
        """Get discovered nodes"""
        return self.known_nodes
```

## 🔒 Security and Authentication

### Distributed System Security

```python
class SecurityManager(Node):
    def __init__(self):
        super().__init__('security_manager')

        # Security configuration
        self.declare_parameter('security_enabled', True)
        self.declare_parameter('certificate_file', 'certs/robot_cert.pem')
        self.declare_parameter('key_file', 'certs/robot_key.pem')
        self.declare_parameter('ca_file', 'certs/ca_cert.pem')

        if not self.get_parameter('security_enabled').value:
            self.get_logger().warn('Security is disabled!')
            return

        # Authentication service
        self.auth_service = self.create_service(
            AuthenticateNode, 'authenticate_node', self.authenticate_callback)

        # Certificate management
        self.certificate_store = CertificateStore(
            cert_file=self.get_parameter('certificate_file').value,
            key_file=self.get_parameter('key_file').value,
            ca_file=self.get_parameter('ca_file').value
        )

        # Access control
        self.access_control = AccessControlManager()

        # Monitoring for security events
        self.security_monitor = SecurityMonitor(self)

        self.get_logger().info('Security Manager initialized')

    def authenticate_callback(self, request, response):
        """Handle node authentication requests"""
        node_id = request.node_id
        certificate = request.certificate
        challenge = request.challenge
        signature = request.signature

        try:
            # Verify certificate
            if not self.certificate_store.verify_certificate(certificate):
                response.success = False
                response.reason = 'Invalid certificate'
                return response

            # Verify challenge signature
            if not self.certificate_store.verify_signature(challenge, signature, certificate):
                response.success = False
                response.reason = 'Invalid signature'
                return response

            # Generate session token
            session_token = self.generate_session_token(node_id)

            response.success = True
            response.session_token = session_token
            response.reason = 'Authentication successful'

            self.get_logger().info(f'Node {node_id} authenticated successfully')

        except Exception as e:
            response.success = False
            response.reason = f'Authentication error: {e}'
            self.get_logger().warn(f'Authentication failed for {node_id}: {e}')

        return response

    def generate_session_token(self, node_id):
        """Generate session token for authenticated node"""
        import jwt
        import time

        payload = {
            'node_id': node_id,
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600  # 1 hour expiration
        }

        return jwt.encode(payload, self.certificate_store.private_key, algorithm='RS256')

# Certificate management utility
class CertificateStore:
    def __init__(self, cert_file, key_file, ca_file):
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_file = ca_file

        self.load_certificates()

    def load_certificates(self):
        """Load certificates and keys"""
        try:
            import ssl

            # Load certificate and key
            self.certificate = ssl.PEM_cert_to_DER_cert(open(self.cert_file).read())
            self.private_key = open(self.key_file).read()

            # Load CA certificate
            self.ca_cert = ssl.PEM_cert_to_DER_cert(open(self.ca_file).read())

        except Exception as e:
            raise Exception(f'Failed to load certificates: {e}')

    def verify_certificate(self, cert_pem):
        """Verify certificate against CA"""
        try:
            # Simplified certificate verification
            # In reality, would use proper certificate chain verification
            return cert_pem.startswith('-----BEGIN CERTIFICATE-----')

        except Exception:
            return False

    def verify_signature(self, challenge, signature, certificate):
        """Verify digital signature"""
        # Simplified signature verification
        # In reality, would use proper cryptographic verification
        return len(signature) > 0 and len(challenge) > 0

# Access control manager
class AccessControlManager:
    def __init__(self):
        # Define access control rules
        self.access_rules = {
            'sensors': ['controller', 'planner', 'safety'],
            'actuators': ['controller', 'safety'],
            'mission_control': ['planner', 'operator'],
            'configuration': ['operator']
        }

    def check_access(self, node_role, resource):
        """Check if node has access to resource"""
        if resource in self.access_rules:
            return node_role in self.access_rules[resource]
        return False
```

---

## 🎯 Best Practices

### Network Optimization

1. **Use Quality of Service**: Match QoS profiles to application requirements
2. **Network Segmentation**: Separate control, sensor, and configuration networks
3. **Bandwidth Management**: Limit message rates and sizes where possible
4. **Redundancy**: Multiple network paths for critical communications
5. **Monitoring**: Continuous network health monitoring and alerts

### System Design Principles

1. **Loose Coupling**: Minimize dependencies between distributed components
2. **Fault Tolerance**: Design for component failures and network issues
3. **Scalability**: Plan for adding more nodes and robots to the system
4. **Security**: Implement authentication, authorization, and encryption
5. **Observability**: Comprehensive logging, monitoring, and debugging tools

### Performance Guidelines

1. **Latency Optimization**: Minimize communication delays for real-time control
2. **Load Balancing**: Distribute computational load appropriately
3. **Data Locality**: Process data close to where it's generated
4. **Caching**: Cache frequently accessed data to reduce network traffic
5. **Batching**: Combine small messages into larger ones when possible

---

## 🎉 Chapter Summary

Distributed systems are essential for scaling modern robotic applications across multiple computers and robots:

1. **Network Architecture**: Proper network design is crucial for system performance
2. **Multi-Robot Coordination**: Enables complex collaborative behaviors
3. **Collaborative Perception**: Combines sensor data from multiple robots
4. **System Monitoring**: Essential for maintaining distributed system health
5. **Security**: Critical for protecting distributed robotic systems

The key to successful distributed robotics is designing robust, secure, and scalable communication architectures that can handle the complexity of multi-computer, multi-robot systems.

**[← Back to Chapter 3: Communication Patterns](03-communication-patterns.md) | [Continue to Chapter 5: Hardware Introduction →](05-hardware-intro.md)**

## Chapter 4 Knowledge Check

### Question 1: Which network type is best for real-time control systems?

**Options:**
- A) WiFi
- B) 5G
- C) Ethernet
- D) CAN bus

**Answer**
> **Correct Answer:** C) Ethernet
>
> Ethernet provides the best combination of high bandwidth (1–10 Gbps) and very low latency (&lt;1ms) for real-time control systems, making it ideal for applications requiring precise timing and deterministic communication.

---

### Question 2: What is the primary advantage of distributed robotics systems?

**Options:**
- A) Lower cost
- B) Simpler programming
- C) Scalability and performance
- D) Reduced complexity

**Answer**
> **Correct Answer:** C) Scalability and performance
>
> Distributed robotics systems allow for computational scaling by using multiple processors, enable physical placement of components where needed, support specialized hardware (GPUs, FPGAs), and provide improved reliability through redundancy.

---

### Question 3: Which DDS configuration parameter controls the communication domain?

**Options:**
- A) ROS_DOMAIN_ID
- B) RMW_IMPLEMENTATION
- C) CYCLONEDDS_URI
- D) NETWORK_INTERFACE

**Answer**
> **Correct Answer:** A) ROS_DOMAIN_ID
>
> The ROS_DOMAIN_ID environment variable controls which DDS domain the ROS 2 system operates in. Nodes in different domains cannot communicate with each other, providing network isolation.

---

### Question 4: What is the purpose of collaborative perception in multi-robot systems?

**Options:**
- A) Reduce battery usage
- B) Combine sensor data from multiple robots
- C) Simplify robot control
- D) Reduce network traffic

**Answer**
> **Correct Answer:** B) Combine sensor data from multiple robots
>
> Collaborative perception enables robots to share and fuse sensor data, creating a more comprehensive understanding of the environment. This allows for better object detection, mapping, and situational awareness than any single robot could achieve alone.