---
title: "12. Sensor Fusion"
sidebar_label: "12. Sensor Fusion"
sidebar_position: 4
---

# Chapter 12: Sensor Fusion

## Creating Robust Perception Through Multi-Modal Sensing

Sensor fusion is the process of combining data from multiple sensors to produce more accurate, reliable, and comprehensive information about the environment than any single sensor could provide alone. In humanoid robotics, sensor fusion is essential for creating robust perception systems that can operate reliably in complex, dynamic environments.

## 🎯 Chapter Overview

### Learning Objectives
By the end of this chapter, you will be able to:
- Understand the principles and benefits of sensor fusion
- Implement Kalman filters for state estimation
- Combine camera, LiDAR, and IMU data for robust perception
- Apply probabilistic approaches to sensor fusion
- Develop real-time sensor fusion pipelines using ROS 2
- Handle sensor uncertainties and error modeling

### Prerequisites
- **Chapter 11**: Computer Vision fundamentals
- Basic understanding of probability and statistics
- Linear algebra fundamentals
- ROS 2 navigation and tf2 concepts
- Python programming skills

### Required Sensors and Hardware
- Camera (RGB or RGB-D)
- Inertial Measurement Unit (IMU)
- LiDAR sensor (recommended but not required)
- Sufficient computational resources for real-time processing

---

## 🔄 12.1 Introduction to Sensor Fusion

### What is Sensor Fusion?

Sensor fusion is the process of integrating data from multiple sensors to produce information that is more accurate, complete, or dependable than could be achieved by using any single sensor individually.

#### **Why Sensor Fusion Matters**

**Complementary Strengths**: Different sensors excel in different conditions
- **Cameras**: High resolution, rich visual information, but sensitive to lighting
- **LiDAR**: Accurate depth measurement, works in darkness, but lower resolution
- **IMU**: High-frequency motion data, but prone to drift over time
- **GPS**: Global positioning, but unavailable indoors and prone to multipath

**Improved Robustness**: Multi-modal systems can compensate for individual sensor failures
**Enhanced Accuracy**: Combining measurements reduces uncertainty and noise
**Redundancy**: Critical for safety-critical humanoid robot applications

### Types of Sensor Fusion

#### **Complementary Fusion**
Sensors measure different aspects of the same phenomenon
- Example: Camera provides appearance, LiDAR provides depth
- Result: Rich 3D understanding with color and texture

#### **Competitive Fusion**
Multiple sensors measure the same quantity
- Example: Multiple cameras for depth estimation
- Result: More accurate measurements through redundancy

#### **Cooperative Fusion**
Sensors work together to measure something neither can measure alone
- Example: Stereo cameras for 3D reconstruction
- Result: Capabilities beyond individual sensor limitations

---

## 📊 12.2 Mathematical Foundations

### Probability and Statistics Basics

#### **Gaussian Distributions**
Most sensor fusion algorithms assume Gaussian noise distributions:

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, rho=0):
    """2D Gaussian probability density function"""
    coeff = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))

    exponent = -1 / (2 * (1 - rho**2)) * (
        ((x - mu_x)**2 / sigma_x**2) +
        ((y - mu_y)**2 / sigma_y**2) -
        (2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y))
    )

    return coeff * np.exp(exponent)

# Example: Visualize sensor uncertainty
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
z = gaussian_2d(x, y, 0, 0, 1, 1.5)

plt.contour(x, y, z, levels=10)
plt.title('2D Gaussian Sensor Uncertainty')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.colorbar(label='Probability Density')
plt.show()
```

#### **Covariance Matrices**
Represent uncertainty and correlations between variables:

```python
def create_position_covariance(x_var, y_var, xy_cov=0):
    """Create covariance matrix for 2D position"""
    return np.array([
        [x_var, xy_cov],
        [xy_cov, y_var]
    ])

# Example uncertainties
camera_position_cov = create_position_covariance(0.1, 0.1, 0.05)
lidar_position_cov = create_position_covariance(0.05, 0.05, 0.01)

print("Camera covariance:")
print(camera_position_cov)
print("\nLiDAR covariance:")
print(lidar_position_cov)
```

---

## 🎛️ 12.3 Kalman Filters

### Introduction to Kalman Filters

The Kalman filter is an algorithm that uses a series of measurements observed over time to produce estimates of unknown variables. It's widely used in robotics for state estimation and sensor fusion.

#### **Kalman Filter Components**

```python
class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        """
        Initialize Kalman Filter

        Args:
            dim_x: State dimension
            dim_z: Measurement dimension
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State vector [x, y, vx, vy] for 2D tracking
        self.x = np.zeros((dim_x, 1))

        # State transition matrix
        self.F = np.eye(dim_x)

        # Measurement matrix
        self.H = np.zeros((dim_z, dim_x))

        # Process noise covariance
        self.Q = np.eye(dim_x) * 0.1

        # Measurement noise covariance
        self.R = np.eye(dim_z) * 1.0

        # Error covariance matrix
        self.P = np.eye(dim_x) * 1.0

    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Update state with measurement"""
        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        """Get current state estimate"""
        return self.x
```

### 2D Position Tracking Example

```python
class PositionTracker:
    def __init__(self):
        # State: [x, y, vx, vy]
        self.kf = KalmanFilter(4, 2)

        # State transition matrix (constant velocity model)
        dt = 0.1  # 100ms time step
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (we only measure position)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance
        self.kf.Q = np.array([
            [0.01, 0, 0.1, 0],
            [0, 0.01, 0, 0.1],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0]
        ])

        # Measurement noise covariance (depends on sensor)
        self.kf.R = np.array([
            [0.1, 0],
            [0, 0.1]
        ])

        self.trajectory = []

    def update(self, measurement):
        """Update filter with new measurement"""
        # Predict
        self.kf.predict()

        # Update with measurement
        z = np.array([[measurement[0]], [measurement[1]]])
        self.kf.update(z)

        # Store trajectory
        state = self.kf.get_state()
        self.trajectory.append(state.flatten())

        return state.flatten()

    def get_velocity(self):
        """Get current velocity estimate"""
        state = self.kf.get_state()
        vx, vy = state[2, 0], state[3, 0]
        return vx, vy
```

### Extended Kalman Filter for Nonlinear Systems

```python
class ExtendedKalmanFilter:
    def __init__(self):
        # State: [x, y, theta, v] (2D pose and velocity)
        self.x = np.array([0, 0, 0, 0])

        # State covariance
        self.P = np.eye(4) * 0.1

        # Process noise covariance
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1])

    def predict(self, u, dt):
        """Predict state with control input"""
        # Control input: [v, omega] (linear and angular velocity)
        v, omega = u

        # Current state
        x, y, theta, vel = self.x

        # Motion model (nonlinear)
        if abs(omega) < 1e-6:
            # Straight line motion
            x_new = x + v * dt * np.cos(theta)
            y_new = y + v * dt * np.sin(theta)
            theta_new = theta
        else:
            # Circular motion
            x_new = x + (v/omega) * (np.sin(theta + omega*dt) - np.sin(theta))
            y_new = y + (v/omega) * (-np.cos(theta + omega*dt) + np.cos(theta))
            theta_new = theta + omega*dt

        vel_new = v

        # Update state
        self.x = np.array([x_new, y_new, theta_new, vel_new])

        # Compute Jacobian of motion model
        if abs(omega) < 1e-6:
            F = np.array([
                [1, 0, -v*dt*np.sin(theta), dt*np.cos(theta)],
                [0, 1, v*dt*np.cos(theta), dt*np.sin(theta)],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            F = np.array([
                [1, 0, (v/omega)*(np.cos(theta + omega*dt) - np.cos(theta)), (1/omega)*(np.sin(theta + omega*dt) - np.sin(theta))],
                [0, 1, (v/omega)*(np.sin(theta + omega*dt) - np.sin(theta)), -(1/omega)*(np.cos(theta + omega*dt) - np.cos(theta))],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement, R):
        """Update state with measurement"""
        # Measurement: [x, y] from GPS/landmarks
        z = measurement[:2]

        # Measurement model
        h = self.x[:2]  # We measure x, y directly

        # Innovation
        y = z - h

        # Measurement Jacobian
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P
```

---

## 🤖 12.4 IMU and Camera Fusion

### IMU Data Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
import numpy as np

class IMUCameraFusion(Node):
    def __init__(self):
        super().__init__('imu_camera_fusion')

        # EKF for pose estimation
        self.ekf = ExtendedKalmanFilter()

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)

        # Publishers
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/fused_pose', 10)

        # IMU bias estimation
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.bias_samples = 0
        self.calibrating = True

        # Vision data
        self.last_visual_update = 0
        self.visual_update_interval = 0.1  # 10 Hz

        self.get_logger().info('IMU-Camera Fusion Node initialized')

    def imu_callback(self, msg):
        """Process IMU measurements"""
        # Extract linear acceleration and angular velocity
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Calibration phase
        if self.calibrating and self.bias_samples < 100:
            self.accel_bias += accel
            self.gyro_bias += gyro
            self.bias_samples += 1
            return
        elif self.calibrating:
            # Finalize calibration
            self.accel_bias /= self.bias_samples
            self.gyro_bias /= self.bias_samples
            self.calibrating = False
            self.get_logger().info('IMU calibration completed')

        # Remove bias
        accel_corrected = accel - self.accel_bias
        gyro_corrected = gyro - self.gyro_bias

        # Convert to control input for EKF
        # Assume constant velocity model with small corrections from IMU
        omega = np.linalg.norm(gyro_corrected[:2])  # Rotation in xy plane
        v = 0.5  # Assume constant forward velocity

        # Predict with EKF
        self.ekf.predict(np.array([v, omega]), dt=0.01)  # 100 Hz IMU

        # Publish fused pose
        self.publish_pose()

    def camera_callback(self, msg):
        """Process camera measurements for visual odometry"""
        current_time = self.get_clock().now()

        # Throttle visual updates
        if (current_time.nanoseconds - self.last_visual_update) < self.visual_update_interval * 1e9:
            return

        self.last_visual_update = current_time.nanoseconds

        # Simulated visual odometry (replace with actual implementation)
        # In practice, you would use feature tracking, SLAM, or AprilTags
        visual_measurement = self.visual_odometry(msg)

        if visual_measurement is not None:
            # Update EKF with visual measurement
            R = np.array([[0.1, 0], [0, 0.1]])  # Visual measurement uncertainty
            self.ekf.update(visual_measurement, R)

    def visual_odometry(self, msg):
        """
        Implement visual odometry using feature tracking
        This is a simplified example - use a proper library like OpenCV's SLAM
        """
        # In practice, you would:
        # 1. Extract features from current frame
        # 2. Match with previous frame
        # 3. Estimate camera motion
        # 4. Return position measurement

        # Simplified simulation
        current_time = self.get_clock().now().nanoseconds
        x = 0.1 * np.sin(current_time * 1e-9)  # Simulated motion
        y = 0.05 * np.cos(current_time * 1e-9)

        return np.array([x, y])

    def publish_pose(self):
        """Publish fused pose estimate"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'

        # Extract position and orientation from EKF state
        x, y, theta, v = self.ekf.x

        pose_msg.pose.pose.position.x = x
        pose_msg.pose.pose.position.y = y
        pose_msg.pose.pose.position.z = 0.0

        # Convert orientation to quaternion
        pose_msg.pose.pose.orientation.w = np.cos(theta/2)
        pose_msg.pose.pose.orientation.x = 0.0
        pose_msg.pose.pose.orientation.y = 0.0
        pose_msg.pose.pose.orientation.z = np.sin(theta/2)

        # Set covariance (simplified)
        pose_msg.pose.covariance[0] = self.ekf.P[0, 0]  # x variance
        pose_msg.pose.covariance[7] = self.ekf.P[1, 1]  # y variance
        pose_msg.pose.covariance[35] = self.ekf.P[2, 2]  # theta variance

        self.pose_pub.publish(pose_msg)
```

### Visual-Inertial Odometry (VIO)

```python
class VisualInertialOdometry:
    def __init__(self):
        # Camera parameters (should be calibrated)
        self.camera_matrix = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ])

        # Feature tracker
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.last_keypoints = None
        self.last_descriptors = None
        self.last_image = None

        # IMU integration
        self.imu_integrator = IMUIntegrator()

        # VIO state
        self.pose = np.eye(4)  # 4x4 transformation matrix
        self.velocity = np.zeros(3)

    def process_frame(self, image, imu_data):
        """Process camera frame with IMU data"""
        if self.last_image is None:
            # Initialize
            self.last_image = image
            self.last_keypoints, self.last_descriptors = self.detect_features(image)
            return

        # Integrate IMU since last frame
        dt = imu_data['timestamp'] - self.imu_integrator.last_timestamp
        self.imu_integrator.integrate(imu_data, dt)

        # Track features
        current_keypoints, current_descriptors = self.detect_features(image)

        # Match features between frames
        matches = self.match_features(
            self.last_descriptors, current_descriptors
        )

        if len(matches) > 10:  # Minimum matches for reliable pose estimation
            # Estimate relative pose
            relative_pose = self.estimate_relative_pose(
                self.last_keypoints, current_keypoints, matches
            )

            # Fuse with IMU prediction
            self.fuse_pose_estimates(relative_pose, self.imu_integrator.get_prediction())

        # Update for next iteration
        self.last_image = image
        self.last_keypoints = current_keypoints
        self.last_descriptors = current_descriptors

    def detect_features(self, image):
        """Detect ORB features in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features between two sets of descriptors"""
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:50]  # Keep top 50 matches

    def estimate_relative_pose(self, kp1, kp2, matches):
        """Estimate relative pose from matched features"""
        # Extract matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Find essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix,
                                     cv2.RANSAC, 0.999, 1.0)

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)

        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        return T

    def fuse_pose_estimates(self, visual_pose, imu_pose):
        """Fuse visual and IMU pose estimates"""
        # Weighted average based on confidences
        visual_weight = 0.7  # Visual measurements are generally more reliable
        imu_weight = 0.3

        # Convert to relative rotation matrices
        R_visual = visual_pose[:3, :3]
        R_imu = imu_pose[:3, :3]

        # Interpolate rotations (simplified)
        R_fused = visual_weight * R_visual + imu_weight * R_imu

        # Normalize rotation matrix
        U, _, Vt = np.linalg.svd(R_fused)
        R_fused = U @ Vt

        # Interpolate translations
        t_fused = visual_weight * visual_pose[:3, 3] + imu_weight * imu_pose[:3, 3]

        # Update pose
        self.pose[:3, :3] = self.pose[:3, :3] @ R_fused
        self.pose[:3, 3] += t_fused

class IMUIntegrator:
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.eye(3)  # Rotation matrix
        self.last_timestamp = None

    def integrate(self, imu_data, dt):
        """Integrate IMU measurements"""
        if self.last_timestamp is None:
            self.last_timestamp = imu_data['timestamp']
            return

        # Extract accelerometer and gyroscope data
        accel = imu_data['linear_acceleration']
        gyro = imu_data['angular_velocity']

        # Update orientation (simplified integration)
        theta = np.linalg.norm(gyro) * dt
        if theta > 1e-6:
            axis = gyro / np.linalg.norm(gyro)
            K = self.skew_symmetric(axis)
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
            self.orientation = self.orientation @ R

        # Transform acceleration to world frame
        accel_world = self.orientation @ accel

        # Update velocity and position
        self.velocity += accel_world * dt
        self.position += self.velocity * dt

        self.last_timestamp = imu_data['timestamp']

    def get_prediction(self):
        """Get current pose prediction"""
        T = np.eye(4)
        T[:3, :3] = self.orientation
        T[:3, 3] = self.position
        return T

    def skew_symmetric(self, v):
        """Create skew-symmetric matrix from vector"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
```

---

## 🌐 12.5 LiDAR and Camera Fusion

### LiDAR Point Cloud Processing

```python
import open3d as o3d
import numpy as np
import cv2

class LidarCameraFusion:
    def __init__(self, camera_matrix, lidar_to_camera_transform):
        self.camera_matrix = camera_matrix
        self.lidar_to_camera = lidar_to_camera_transform

    def project_lidar_to_image(self, point_cloud):
        """Project 3D LiDAR points to 2D image coordinates"""
        # Transform LiDAR points to camera frame
        transformed_points = self.lidar_to_camera @ point_cloud.T

        # Project to image plane
        image_points = self.camera_matrix @ transformed_points[:3, :]

        # Convert to homogeneous coordinates
        image_points = image_points[:2, :] / image_points[2, :]

        return image_points.T, transformed_points[2, :]  # Return 2D points and depths

    def fuse_camera_and_lidar(self, image, point_cloud):
        """Fuse camera image with LiDAR point cloud"""
        # Project LiDAR points to image
        image_points, depths = self.project_lidar_to_image(point_cloud)

        # Filter points within image bounds
        height, width = image.shape[:2]
        mask = (
            (image_points[:, 0] >= 0) & (image_points[:, 0] < width) &
            (image_points[:, 1] >= 0) & (image_points[:, 1] < height) &
            (depths > 0) & (depths < 50)  # Reasonable depth range
        )

        valid_points = image_points[mask]
        valid_depths = depths[mask]

        # Create depth image from LiDAR
        depth_image = np.zeros((height, width))
        for i, (x, y) in enumerate(valid_points):
            x, y = int(x), int(y)
            if depth_image[y, x] == 0 or valid_depths[i] < depth_image[y, x]:
                depth_image[y, x] = valid_depths[i]

        # Fill gaps (optional)
        depth_image = cv2.medianBlur(depth_image.astype(np.float32), 5)

        return depth_image, valid_points, valid_depths

    def create_colored_point_cloud(self, image, point_cloud):
        """Create RGB-colored point cloud from camera and LiDAR data"""
        depth_image, valid_points, valid_depths = self.fuse_camera_and_lidar(image, point_cloud)

        # Get colors for valid points
        colors = []
        for x, y in valid_points.astype(int):
            color = image[y, x]  # BGR format
            colors.append(color[::-1] / 255.0)  # Convert to RGB and normalize

        # Filter original point cloud
        valid_mask = mask & (depths > 0) & (depths < 50)
        colored_points = point_cloud[valid_mask]

        # Create Open3D point cloud
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(colored_points)
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(colors)

        return o3d_point_cloud
```

### Multi-Modal Object Detection

```python
class MultiModalObjectDetector:
    def __init__(self, camera_matrix, lidar_to_camera):
        self.camera_matrix = camera_matrix
        self.lidar_to_camera = lidar_to_camera

        # Object detector (could be YOLO, SSD, etc.)
        self.object_detector = self.load_detector()

        # LiDAR clustering
        self.cluster_threshold = 0.5  # meters
        self.min_cluster_size = 10

    def load_detector(self):
        """Load pre-trained object detector"""
        # This would typically load a trained neural network
        # For example: YOLO, SSD, Faster R-CNN
        pass

    def detect_objects_multi_modal(self, image, point_cloud):
        """Detect objects using both camera and LiDAR"""
        # Camera-based detection
        camera_detections = self.detect_objects_camera(image)

        # LiDAR-based clustering
        lidar_clusters = self.cluster_lidar_points(point_cloud)

        # Fuse detections
        fused_detections = self.fuse_detections(camera_detections, lidar_clusters)

        return fused_detections

    def detect_objects_camera(self, image):
        """Detect objects using camera only"""
        # This would use the loaded object detector
        # Return bounding boxes, classes, and confidence scores

        detections = []

        # Simulated detection for example
        # In practice, you would use your trained model here
        height, width = image.shape[:2]

        # Example detection: person at center
        detections.append({
            'bbox': [width//2 - 50, height//2 - 100, 100, 200],  # [x, y, w, h]
            'class': 'person',
            'confidence': 0.85,
            'source': 'camera'
        })

        return detections

    def cluster_lidar_points(self, point_cloud):
        """Cluster LiDAR points to detect objects"""
        # Convert to numpy array if needed
        if not isinstance(point_cloud, np.ndarray):
            points = np.asarray(point_cloud.points)
        else:
            points = point_cloud

        # Simple clustering based on distance
        clusters = []
        used_indices = set()

        for i, point in enumerate(points):
            if i in used_indices:
                continue

            # Find nearby points
            distances = np.linalg.norm(points - point, axis=1)
            cluster_indices = np.where((distances < self.cluster_threshold) & (distances > 0.01))[0]

            if len(cluster_indices) >= self.min_cluster_size:
                cluster_points = points[cluster_indices]
                clusters.append(cluster_points)
                used_indices.update(cluster_indices)

        # Analyze clusters
        object_clusters = []
        for cluster_points in clusters:
            cluster_info = self.analyze_cluster(cluster_points)
            object_clusters.append(cluster_info)

        return object_clusters

    def analyze_cluster(self, cluster_points):
        """Analyze LiDAR cluster to extract object properties"""
        # Calculate cluster properties
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        max_distance = np.max(distances)

        # Estimate object size
        min_coords = np.min(cluster_points, axis=0)
        max_coords = np.max(cluster_points, axis=0)
        dimensions = max_coords - min_coords

        # Classify object based on size and shape
        object_class = self.classify_lidar_object(dimensions, len(cluster_points))

        return {
            'position': centroid,
            'dimensions': dimensions,
            'class': object_class,
            'confidence': 0.7,  # LiDAR confidence
            'source': 'lidar'
        }

    def classify_lidar_object(self, dimensions, num_points):
        """Classify object based on LiDAR cluster properties"""
        width, length, height = dimensions
        volume = width * length * height

        # Simple classification rules
        if height > 1.5:  # Tall objects
            return 'person'
        elif volume > 2.0:  # Large objects
            return 'vehicle'
        elif volume > 0.5:
            return 'obstacle'
        else:
            return 'unknown'

    def fuse_detections(self, camera_detections, lidar_clusters):
        """Fuse camera and LiDAR detections"""
        fused_objects = []

        # Project LiDAR clusters to image space
        for lidar_obj in lidar_clusters:
            # Project 3D position to 2D image
            point_3d = np.append(lidar_obj['position'], 1)
            point_2d = self.camera_matrix @ (self.lidar_to_camera @ point_3d)[:3]
            point_2d = point_2d[:2] / point_2d[2]

            lidar_obj['image_position'] = point_2d

        # Match camera and LiDAR detections
        matched_pairs = self.match_detections(camera_detections, lidar_clusters)

        # Create fused objects
        for cam_det, lidar_obj in matched_pairs:
            fused_obj = {
                'position': lidar_obj['position'],
                'dimensions': lidar_obj['dimensions'],
                'bbox': cam_det['bbox'],
                'class': cam_det['class'],  # Camera classification is usually more accurate
                'confidence': (cam_det['confidence'] + lidar_obj['confidence']) / 2,
                'sources': ['camera', 'lidar']
            }
            fused_objects.append(fused_obj)

        # Add unmatched detections
        for cam_det in camera_detections:
            if not any(cam_det == pair[0] for pair in matched_pairs):
                fused_obj = {
                    'bbox': cam_det['bbox'],
                    'class': cam_det['class'],
                    'confidence': cam_det['confidence'],
                    'sources': ['camera']
                }
                fused_objects.append(fused_obj)

        for lidar_obj in lidar_clusters:
            if not any(lidar_obj == pair[1] for pair in matched_pairs):
                fused_obj = {
                    'position': lidar_obj['position'],
                    'dimensions': lidar_obj['dimensions'],
                    'class': lidar_obj['class'],
                    'confidence': lidar_obj['confidence'],
                    'sources': ['lidar']
                }
                fused_objects.append(fused_obj)

        return fused_objects

    def match_detections(self, camera_detections, lidar_clusters, threshold=50):
        """Match camera detections with LiDAR clusters"""
        matches = []

        for cam_det in camera_detections:
            bbox_center_x = cam_det['bbox'][0] + cam_det['bbox'][2] / 2
            bbox_center_y = cam_det['bbox'][1] + cam_det['bbox'][3] / 2

            best_match = None
            min_distance = float('inf')

            for lidar_obj in lidar_clusters:
                img_pos = lidar_obj.get('image_position')
                if img_pos is not None:
                    distance = np.sqrt(
                        (bbox_center_x - img_pos[0])**2 +
                        (bbox_center_y - img_pos[1])**2
                    )

                    if distance < min_distance and distance < threshold:
                        min_distance = distance
                        best_match = lidar_obj

            if best_match:
                matches.append((cam_det, best_match))

        return matches
```

---

## ⚡ 12.6 Real-Time Sensor Fusion Implementation

### Multi-Threaded Sensor Processing

```python
import threading
import queue
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped

class RealTimeSensorFusion(Node):
    def __init__(self):
        super().__init__('real_time_sensor_fusion')

        # Thread-safe queues for sensor data
        self.image_queue = queue.Queue(maxsize=5)
        self.imu_queue = queue.Queue(maxsize=50)  # Higher rate for IMU
        self.lidar_queue = queue.Queue(maxsize=10)

        # Fusion algorithms
        self.imu_camera_fusion = IMUCameraFusion()
        self.lidar_camera_fusion = LidarCameraFusion(
            camera_matrix=self.get_camera_matrix(),
            lidar_to_camera=self.get_lidar_to_camera_transform()
        )

        # Synchronization
        self.latest_sensor_data = {}
        self.fusion_lock = threading.Lock()

        # Publishers
        self.fused_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10
        )

        # Processing threads
        self.running = True
        self.threads = []

        # Start processing threads
        self.start_threads()

        self.get_logger().info('Real-time Sensor Fusion Node started')

    def start_threads(self):
        """Start sensor processing threads"""
        # Camera processing thread
        camera_thread = threading.Thread(target=self.camera_processing_loop)
        camera_thread.daemon = True
        self.threads.append(camera_thread)

        # IMU processing thread
        imu_thread = threading.Thread(target=self.imu_processing_loop)
        imu_thread.daemon = True
        self.threads.append(imu_thread)

        # LiDAR processing thread
        lidar_thread = threading.Thread(target=self.lidar_processing_loop)
        lidar_thread.daemon = True
        self.threads.append(lidar_thread)

        # Fusion thread
        fusion_thread = threading.Thread(target=self.fusion_loop)
        fusion_thread.daemon = True
        self.threads.append(fusion_thread)

        # Start all threads
        for thread in self.threads:
            thread.start()

    def camera_callback(self, msg):
        """Camera image callback"""
        try:
            if not self.image_queue.full():
                self.image_queue.put(msg)
        except queue.Full:
            # Drop oldest frame
            try:
                self.image_queue.get_nowait()
                self.image_queue.put(msg)
            except queue.Empty:
                pass

    def imu_callback(self, msg):
        """IMU data callback"""
        try:
            if not self.imu_queue.full():
                self.imu_queue.put(msg)
        except queue.Full:
            # Drop oldest IMU data
            try:
                self.imu_queue.get_nowait()
                self.imu_queue.put(msg)
            except queue.Empty:
                pass

    def lidar_callback(self, msg):
        """LiDAR scan callback"""
        try:
            if not self.lidar_queue.full():
                self.lidar_queue.put(msg)
        except queue.Full:
            # Drop oldest scan
            try:
                self.lidar_queue.get_nowait()
                self.lidar_queue.put(msg)
            except queue.Empty:
                pass

    def camera_processing_loop(self):
        """Process camera images in separate thread"""
        while self.running:
            try:
                # Wait for camera data (timeout to allow clean shutdown)
                image_msg = self.image_queue.get(timeout=0.1)

                # Process image
                start_time = time.time()

                # Convert to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')

                # Object detection (example)
                detections = self.detect_objects(cv_image)

                # Store with timestamp
                with self.fusion_lock:
                    self.latest_sensor_data['camera'] = {
                        'timestamp': image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9,
                        'image': cv_image,
                        'detections': detections,
                        'processing_time': time.time() - start_time
                    }

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Camera processing error: {e}')

    def imu_processing_loop(self):
        """Process IMU data in separate thread"""
        while self.running:
            try:
                # Wait for IMU data
                imu_msg = self.imu_queue.get(timeout=0.01)  # Higher rate

                # Process IMU data
                imu_data = {
                    'linear_acceleration': np.array([
                        imu_msg.linear_acceleration.x,
                        imu_msg.linear_acceleration.y,
                        imu_msg.linear_acceleration.z
                    ]),
                    'angular_velocity': np.array([
                        imu_msg.angular_velocity.x,
                        imu_msg.angular_velocity.y,
                        imu_msg.angular_velocity.z
                    ]),
                    'timestamp': imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9
                }

                # Store with timestamp
                with self.fusion_lock:
                    self.latest_sensor_data['imu'] = imu_data

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'IMU processing error: {e}')

    def lidar_processing_loop(self):
        """Process LiDAR data in separate thread"""
        while self.running:
            try:
                # Wait for LiDAR data
                lidar_msg = self.lidar_queue.get(timeout=0.05)

                # Convert LiDAR scan to point cloud
                point_cloud = self.lidar_scan_to_point_cloud(lidar_msg)

                # Store with timestamp
                with self.fusion_lock:
                    self.latest_sensor_data['lidar'] = {
                        'timestamp': lidar_msg.header.stamp.sec + lidar_msg.header.stamp.nanosec * 1e-9,
                        'point_cloud': point_cloud
                    }

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'LiDAR processing error: {e}')

    def fusion_loop(self):
        """Main fusion loop running at target frequency"""
        target_frequency = 30  # Hz
        target_period = 1.0 / target_frequency

        while self.running:
            start_time = time.time()

            try:
                with self.fusion_lock:
                    sensor_data = dict(self.latest_sensor_data)

                # Perform fusion
                fused_pose = self.perform_fusion(sensor_data)

                # Publish result
                if fused_pose is not None:
                    self.publish_fused_pose(fused_pose)

            except Exception as e:
                self.get_logger().error(f'Fusion error: {e}')

            # Maintain target frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, target_period - elapsed)
            time.sleep(sleep_time)

    def perform_fusion(self, sensor_data):
        """Perform sensor fusion"""
        if 'imu' not in sensor_data:
            return None

        current_time = time.time()

        # Time-synchronized fusion
        if 'camera' in sensor_data:
            # Check time synchronization
            time_diff = abs(sensor_data['camera']['timestamp'] - sensor_data['imu']['timestamp'])
            if time_diff < 0.1:  # Within 100ms
                # Perform camera-IMU fusion
                pose = self.imu_camera_fusion.get_state() if hasattr(self.imu_camera_fusion, 'get_state') else None
                return pose

        if 'lidar' in sensor_data:
            time_diff = abs(sensor_data['lidar']['timestamp'] - sensor_data['imu']['timestamp'])
            if time_diff < 0.1:  # Within 100ms
                # Perform LiDAR-IMU fusion
                # This would involve more complex algorithms
                pass

        return None

    def lidar_scan_to_point_cloud(self, scan_msg):
        """Convert ROS 2 LaserScan to point cloud"""
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))

        points = []
        for i, (angle, range_val) in enumerate(zip(angles, scan_msg.ranges)):
            if scan_msg.range_min <= range_val <= scan_msg.range_max:
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                z = 0.0  # Assuming 2D LiDAR
                points.append([x, y, z])

        return np.array(points)

    def publish_fused_pose(self, pose):
        """Publish fused pose estimate"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'

        # Convert pose to ROS 2 message format
        if len(pose) >= 3:
            pose_msg.pose.pose.position.x = pose[0]
            pose_msg.pose.pose.position.y = pose[1]
            pose_msg.pose.pose.position.z = 0.0

        self.fused_pose_pub.publish(pose_msg)

    def get_camera_matrix(self):
        """Get camera intrinsic matrix"""
        # This should be calibrated for your specific camera
        return np.array([
            [525.0, 0.0, 320.0],
            [0.0, 525.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

    def get_lidar_to_camera_transform(self):
        """Get transformation from LiDAR to camera frame"""
        # This should be calibrated for your specific setup
        T = np.eye(4)
        T[:3, :3] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        T[:3, 3] = np.array([0.1, 0.0, 0.2])  # Translation

        return T

    def __del__(self):
        """Cleanup"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
```

---

## 🧪 12.7 Practical Project: Multi-Sensor Navigation System

### Project Overview
Create a comprehensive navigation system that fuses camera, IMU, and LiDAR data to enable robust humanoid robot navigation in complex environments.

### System Architecture

```python
class MultiSensorNavigationSystem:
    def __init__(self):
        # Sensor fusion components
        self.imu_camera_fusion = IMUCameraFusion()
        self.lidar_camera_fusion = LidarCameraFusion()
        self.object_detector = MultiModalObjectDetector()

        # Navigation components
        self.path_planner = PathPlanner()
        self.obstacle_avoidance = ObstacleAvoidance()

        # State estimation
        self.robot_pose = np.eye(4)
        self.robot_velocity = np.zeros(3)

        # Mapping and localization
        self.occupancy_map = OccupancyMap(resolution=0.05)
        self.particle_filter = ParticleFilter()

    def navigation_loop(self, sensor_data):
        """Main navigation loop"""
        # 1. Update state estimation
        self.update_pose_estimation(sensor_data)

        # 2. Detect objects and obstacles
        objects = self.object_detector.detect_objects_multi_modal(
            sensor_data['camera'], sensor_data['lidar']
        )

        # 3. Update occupancy map
        self.update_map(sensor_data['lidar'], self.robot_pose)

        # 4. Localize robot
        self.localize_robot(sensor_data)

        # 5. Plan path
        path = self.path_planner.plan_path(
            self.robot_pose, self.occupancy_map
        )

        # 6. Avoid obstacles
        velocity_command = self.obstacle_avoidance.compute_command(
            path, objects, self.robot_pose
        )

        return velocity_command

    def update_pose_estimation(self, sensor_data):
        """Update robot pose using sensor fusion"""
        if 'imu' in sensor_data and 'camera' in sensor_data:
            # High-frequency IMU prediction
            self.imu_camera_fusion.imu_callback(sensor_data['imu'])

            # Low-frequency camera correction
            self.imu_camera_fusion.camera_callback(sensor_data['camera'])

            # Get updated pose
            pose_msg = self.imu_camera_fusion.get_latest_pose()
            if pose_msg:
                self.robot_pose = self.ros_pose_to_matrix(pose_msg.pose)

    def update_map(self, point_cloud, robot_pose):
        """Update occupancy map with LiDAR data"""
        # Transform point cloud to world frame
        world_points = robot_pose @ point_cloud.T

        # Update occupancy map
        for point in world_points.T:
            if np.linalg.norm(point[:2]) < 10.0:  # Within range
                grid_x, grid_y = self.occupancy_map.world_to_grid(point[0], point[1])
                self.occupancy_map.update_cell(grid_x, grid_y, occupied=True)

    def localize_robot(self, sensor_data):
        """Localize robot using particle filter"""
        if 'camera' in sensor_data:
            # Visual landmarks for localization
            landmarks = self.detect_landmarks(sensor_data['camera'])
            self.particle_filter.update_with_visual_measurements(landmarks)

        if 'imu' in sensor_data:
            # IMU motion prediction
            self.particle_filter.predict(sensor_data['imu'])

        # Get best pose estimate
        self.robot_pose = self.particle_filter.get_best_pose()
```

---

## ✅ 12.8 Chapter Summary and Key Takeaways

### Core Concepts Covered
1. **Sensor Fusion Principles**: Benefits and types of multi-sensor integration
2. **Mathematical Foundations**: Probability, statistics, and uncertainty modeling
3. **Kalman Filters**: Basic and Extended Kalman filters for state estimation
4. **IMU-Camera Fusion**: Visual-inertial odometry and motion estimation
5. **LiDAR-Camera Fusion**: Multi-modal object detection and 3D perception
6. **Real-time Processing**: Multi-threaded sensor processing architectures
7. **Practical Applications**: Navigation systems and robust perception

### Key Skills Developed
- Implementing Kalman filters for state estimation
- Fusing different sensor modalities for improved perception
- Handling sensor uncertainties and error modeling
- Building real-time multi-sensor processing pipelines
- Creating robust navigation systems

### Common Challenges and Solutions
- **Sensor Synchronization**: Time alignment and calibration issues
- **Different Update Rates**: Multi-rate sensor fusion techniques
- **Sensor Failures**: Redundancy and fault-tolerant designs
- **Computational Complexity**: Efficient algorithms and parallel processing

### Best Practices
- **Modular Design**: Separate sensor interfaces, fusion algorithms, and applications
- **Error Modeling**: Accurate covariance matrices and uncertainty quantification
- **Calibration**: Regular sensor calibration and validation
- **Testing**: Comprehensive testing with simulated and real-world data

---

## 🚀 Next Steps

In the next chapter, we'll explore **Perception Algorithms** (Chapter 13), where we'll dive deeper into advanced perception techniques including deep learning approaches, semantic understanding, and context-aware robotics.

### Preparation for Next Chapter
- Practice implementing different Kalman filter variants
- Experiment with real sensor data (if available)
- Study basic deep learning concepts for computer vision
- Review machine learning fundamentals for perception

**Remember**: Sensor fusion is the key to creating robust, reliable humanoid robots. The principles and techniques covered in this chapter form the foundation for advanced perception systems that can operate safely and effectively in real-world environments! 🤖📊🔄