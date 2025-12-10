---
title: "11. Computer Vision"
sidebar_label: "11. Computer Vision"
sidebar_position: 3
---

# Chapter 11: Computer Vision

## Enabling Robots to See and Understand the World

Computer vision is the cornerstone of modern humanoid robotics, providing robots with the ability to perceive, interpret, and interact with their environment through visual information. This chapter covers the fundamental concepts, practical implementations, and advanced techniques for building robust computer vision systems for humanoid robots.

## 🎯 Chapter Overview

### Learning Objectives
By the end of this chapter, you will be able to:
- Set up and configure computer vision pipelines using OpenCV
- Implement real-time image processing for robotics applications
- Develop object detection and tracking systems
- Create 3D vision systems for spatial understanding
- Integrate computer vision with ROS 2 for robot control
- Optimize vision algorithms for real-time performance

### Prerequisites
- Basic Python programming skills
- Understanding of linear algebra and matrix operations
- ROS 2 fundamentals from Quarter 1
- Familiarity with Linux command line

### Hardware Requirements
- USB or integrated camera (webcam or industrial camera)
- Dedicated GPU recommended for real-time processing
- Sufficient RAM (8GB minimum, 16GB recommended)

### 🔗 Skill Bridge for Different Backgrounds

#### **👨‍💻 For Software Engineers**
- **Familiar Concept**: Image processing algorithms are similar to data transformation pipelines
- **Bridge**: Think of computer vision as a specialized data processing pipeline where images are the input data type
- **Learning Focus**: Real-time processing constraints and hardware-specific optimizations

#### **🤖 For Computer Scientists**
- **Familiar Concept**: Machine learning algorithms and optimization techniques
- **Bridge**: Apply algorithmic complexity analysis to real-time vision systems
- **Learning Focus**: Physical constraints and practical implementation challenges

#### **📊 For Data Scientists**
- **Familiar Concept**: Pattern recognition and statistical analysis
- **Bridge**: Treat images as high-dimensional data for classification and analysis
- **Learning Focus**: Real-time streaming analytics and adaptive systems

#### **⚙️ For Mechanical Engineers**
- **Familiar Concept**: Spatial relationships and coordinate systems
- **Bridge**: Visualize computer vision as optical measurement systems with digital processing
- **Learning Focus**: Camera calibration and 3D geometry transformation

---

## 📷 11.1 Introduction to Computer Vision for Robotics

### What is Computer Vision?
Computer vision is the field of artificial intelligence that enables computers and systems to derive meaningful information from digital images, videos, and other visual inputs. In robotics, computer vision serves as the primary sensing modality for understanding the environment.

### Why Computer Vision Matters in Humanoid Robotics

#### **Environmental Perception**
- **Object Recognition**: Identify and classify objects in the environment
- **Scene Understanding**: Comprehend spatial relationships and context
- **Navigation**: Path planning and obstacle avoidance
- **Human Interaction**: Face detection, gesture recognition, and emotion analysis

#### **Task Execution**
- **Manipulation**: Visual servoing for precise object handling
- **Quality Control**: Inspection and verification tasks
- **Safety**: Monitoring for hazardous conditions
- **Communication**: Reading text, symbols, and visual instructions

### Types of Computer Vision Tasks

#### **Low-Level Vision**
- Image preprocessing and enhancement
- Edge detection and feature extraction
- Image segmentation and region analysis
- Optical flow and motion detection

#### **Mid-Level Vision**
- 3D reconstruction from multiple views
- Object detection and recognition
- Shape analysis and description
- Stereo vision and depth estimation

#### **High-Level Vision**
- Scene understanding and interpretation
- Activity recognition and behavior analysis
- Visual reasoning and decision making
- Human-robot interaction

---

## 🛠️ 11.2 Setting Up Your Computer Vision Environment

### OpenCV Installation and Configuration

#### **Ubuntu/Debian Installation**
```bash
# Update package manager
sudo apt update

# Install OpenCV development libraries
sudo apt install -y python3-opencv
sudo apt install -y libopencv-dev python3-dev
sudo apt install -y build-essential cmake pkg-config

# Install additional Python packages
pip3 install numpy matplotlib scikit-image
pip3 install opencv-python opencv-contrib-python
```

#### **Verify Installation**
```python
# test_opencv.py
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# Test basic functionality
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera access: OK")
    cap.release()
else:
    print("Camera access: FAILED")
```

### Camera Setup and Calibration

#### **Camera Access with ROS 2**
```python
# camera_node.py
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(0.033, self.capture_callback)  # 30 FPS

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def capture_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert OpenCV image to ROS 2 message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(msg)
        else:
            self.get_logger().error('Failed to capture frame')

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### **Camera Calibration**
```python
# camera_calibration.py
import cv2
import numpy as np
import glob

def calibrate_camera():
    # Prepare object points
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points
    objpoints = []
    imgpoints = []

    # Load calibration images
    images = glob.glob('calibration_images/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Save calibration parameters
    np.savez('camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    return ret, mtx, dist

if __name__ == '__main__':
    ret, mtx, dist = calibrate_camera()
    if ret:
        print("Camera calibration successful!")
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients: {dist}")
    else:
        print("Camera calibration failed!")
```

---

## 🔍 11.3 Image Processing Fundamentals

### Basic Image Operations

#### **Image Reading and Display**
```python
import cv2
import numpy as np

# Read image
image = cv2.imread('robot_workshop.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Display images
cv2.imshow('Original', image)
cv2.imshow('Grayscale', gray)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### **Edge Detection**
```python
def detect_edges(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    return edges

# Example usage
image = cv2.imread('objects.jpg')
edges = detect_edges(image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Feature Detection and Description

#### **SIFT (Scale-Invariant Feature Transform)**
```python
def detect_sift_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints on image
    image_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return image_with_keypoints, keypoints, descriptors

# Example usage
image = cv2.imread('scenary.jpg')
result, kp, desc = detect_sift_features(image)
cv2.imshow('SIFT Features', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### **ORB (Oriented FAST and Rotated BRIEF)**
```python
def detect_orb_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw keypoints on image
    image_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, color=(0, 255, 0), flags=0
    )

    return image_with_keypoints, keypoints, descriptors
```

---

## 🎯 11.4 Object Detection and Tracking

### Template Matching
```python
def template_matching(main_image, template):
    # Convert images to grayscale
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Get template dimensions
    w, h = template_gray.shape[::-1]

    # Perform template matching
    result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find location of best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw rectangle around matched region
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(main_image, top_left, bottom_right, (0, 255, 0), 2)

    return main_image, max_loc, max_val
```

### Color-Based Object Detection
```python
def detect_by_color(image, lower_color, upper_color):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask for specified color range
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, contours

# Example usage - detect red objects
image = cv2.imread('colored_objects.jpg')
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
result, contours = detect_by_color(image, lower_red, upper_red)
```

### Optical Flow and Motion Tracking
```python
def track_objects_optical_flow(video_path):
    # Read video
    cap = cv2.VideoCapture(video_path)

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read first frame and find corners
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('Optical Flow', img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()
```

---

## 🤖 11.5 Computer Vision for Humanoid Robots

### Human Detection and Tracking
```python
import cv2
import numpy as np

class HumanDetector:
    def __init__(self):
        # Load pre-trained HOG detector for people
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_humans(self, image):
        # Detect people in the image
        boxes, weights = self.hog.detectMultiScale(
            image, winStride=(8, 8), padding=(32, 32), scale=1.05
        )

        # Draw bounding boxes around detected humans
        for i, (x, y, w, h) in enumerate(boxes):
            if weights[i] > 0.5:  # Confidence threshold
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f'Human {weights[i]:.2f}',
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)

        return image, boxes, weights

# Usage example
detector = HumanDetector()
image = cv2.imread('people_scene.jpg')
result, boxes, weights = detector.detect_humans(image)
cv2.imshow('Human Detection', result)
cv2.waitKey(0)
```

### Face Detection and Recognition
```python
class FaceRecognitionSystem:
    def __init__(self):
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Load face recognition model (using LBPH)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]

            # Recognize face (if trained)
            try:
                label, confidence = self.recognizer.predict(face_roi)
                name = f"Person {label}" if confidence < 100 else "Unknown"
                cv2.putText(image, name, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            except:
                cv2.putText(image, "Unknown", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return image, faces
```

### Gesture Recognition
```python
class GestureDetector:
    def __init__(self):
        # Skin color range for hand detection
        self.lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def detect_hand(self, image):
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for skin color
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (hand)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            # Get convex hull and defects for gesture recognition
            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull)

            # Count fingers (simplified approach)
            finger_count = self.count_fingers(defects)

            return max_contour, finger_count

        return None, 0

    def count_fingers(self, defects):
        if defects is None:
            return 0

        count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            # Filter defects based on depth and angle
            if d > 10000:  # Depth threshold
                count += 1

        return min(count + 1, 5)  # Ensure max 5 fingers
```

---

## 🌐 11.6 3D Vision and Depth Perception

### Stereo Vision Setup
```python
class StereoVisionSystem:
    def __init__(self, camera_matrix1, camera_matrix2, dist_coeffs1, dist_coeffs2, R, T):
        self.camera_matrix1 = camera_matrix1
        self.camera_matrix2 = camera_matrix2
        self.dist_coeffs1 = dist_coeffs1
        self.dist_coeffs2 = dist_coeffs2
        self.R = R  # Rotation matrix
        self.T = T  # Translation vector

        # Compute rectification transforms
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2,
            (640, 480), R, T, alpha=0.9
        )

        self.R1, self.R2 = R1, R2
        self.P1, self.P2 = P1, P2
        self.Q = Q  # Disparity-to-depth mapping matrix

    def compute_disparity(self, left_img, right_img):
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Stereo BM matcher
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
        disparity = stereo.compute(left_gray, right_gray)

        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(
            disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        return disparity, disparity_normalized

    def compute_depth_map(self, disparity):
        # Convert disparity to 3D points
        points_3D = cv2.reprojectImageTo3D(disparity, self.Q)
        return points_3D
```

### Point Cloud Generation
```python
def generate_point_cloud(disparity, Q, image):
    # Reproject disparity to 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # Get valid points (where disparity != 0)
    mask = disparity > disparity.min()
    points = points_3D[mask]
    colors = image[mask]

    # Filter points based on distance
    valid_depth = (points[:, 2] > 0.1) & (points[:, 2] < 10.0)
    points = points[valid_depth]
    colors = colors[valid_depth]

    return points, colors

# Visualization with matplotlib
def visualize_point_cloud(points, colors):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize colors for visualization
    colors_norm = colors.astype(float) / 255.0

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
              c=colors_norm, s=1, marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    plt.show()
```

---

## 🚀 11.7 Real-Time Performance Optimization

### Multi-threading for Video Processing
```python
import threading
import queue
import time

class VideoProcessor:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False

    def capture_frames(self):
        """Capture frames in separate thread"""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Drop oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except queue.Empty:
                        pass

        cap.release()

    def process_frames(self):
        """Process frames in separate thread"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)

                # Apply computer vision processing
                processed_frame = self.process_single_frame(frame)

                if not self.result_queue.full():
                    self.result_queue.put(processed_frame)
                else:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(processed_frame)
                    except queue.Empty:
                        pass

            except queue.Empty:
                continue

    def process_single_frame(self, frame):
        """Process a single frame"""
        # Example: edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Convert back to BGR for display
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return edges_bgr

    def start(self):
        """Start video processing"""
        self.running = True

        # Start threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()

    def stop(self):
        """Stop video processing"""
        self.running = False

        # Wait for threads to finish
        self.capture_thread.join()
        self.process_thread.join()

    def get_latest_frame(self):
        """Get the latest processed frame"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

# Usage example
processor = VideoProcessor(0)
processor.start()

try:
    while True:
        frame = processor.get_latest_frame()
        if frame is not None:
            cv2.imshow('Processed Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    processor.stop()
    cv2.destroyAllWindows()
```

### GPU Acceleration with CUDA
```python
def setup_cuda():
    """Check and setup CUDA for GPU acceleration"""
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA is available for GPU acceleration")
        return True
    else:
        print("CUDA is not available, using CPU")
        return False

def gpu_accelerated_processing(image):
    """Example of GPU-accelerated image processing"""
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # Upload image to GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        # Convert to grayscale on GPU
        gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur on GPU
        gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)

        # Download result back to CPU
        result = gpu_blurred.download()
        return result
    else:
        # Fallback to CPU processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
```

---

## 🔧 11.8 Integration with ROS 2

### Vision Node for Object Detection
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # CvBridge for ROS 2 - OpenCV conversion
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Image, '/detection/image', 10
        )
        self.object_pos_pub = self.create_publisher(
            PointStamped, '/detected_object/position', 10
        )

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None

        # Object detector
        self.detector = HumanDetector()

        self.get_logger().info('Object Detection Node initialized')

    def camera_info_callback(self, msg):
        """Store camera calibration parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS 2 image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Detect objects
            result_image, boxes, weights = self.detector.detect_humans(cv_image)

            # Calculate 3D position if camera parameters are available
            if self.camera_matrix is not None and len(boxes) > 0:
                self.calculate_object_position(boxes[0], msg.header)

            # Convert back to ROS 2 format and publish
            result_msg = self.bridge.cv2_to_imgmsg(result_image, 'bgr8')
            result_msg.header = msg.header
            self.detection_pub.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def calculate_object_position(self, box, header):
        """Calculate 3D position of detected object"""
        # Assuming objects are on a known plane (e.g., floor)
        # This is a simplified calculation - real implementation would need
        # depth information or additional assumptions

        x, y, w, h = box
        center_x = x + w // 2
        center_y = y + h // 2

        # Project to 3D (simplified)
        if self.camera_matrix is not None:
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

            # Assuming object height is known (e.g., 1.7m for humans)
            real_height = 1.7
            pixel_height = h

            if pixel_height > 0:
                distance = (real_height * fy) / pixel_height

                # Calculate 3D position
                world_x = (center_x - cx) * distance / fx
                world_y = (center_y - cy) * distance / fy
                world_z = distance

                # Create and publish position message
                pos_msg = PointStamped()
                pos_msg.header = header
                pos_msg.point.x = world_x
                pos_msg.point.y = world_y
                pos_msg.point.z = world_z

                self.object_pos_pub.publish(pos_msg)

    def __del__(self):
        """Cleanup"""
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Visual Servoing Control
```python
class VisualServoingController:
    def __init__(self):
        self.target_position = None
        self.current_position = None
        self.control_gain = 0.5

    def set_target(self, image_point):
        """Set target position in image coordinates"""
        self.target_position = image_point

    def update_current_position(self, image_point):
        """Update current object position"""
        self.current_position = image_point

    def compute_control_command(self):
        """Compute velocity command for visual servoing"""
        if self.target_position is None or self.current_position is None:
            return 0.0, 0.0

        # Calculate error
        error_x = self.target_position[0] - self.current_position[0]
        error_y = self.target_position[1] - self.current_position[1]

        # Proportional control
        vx = self.control_gain * error_x
        vy = self.control_gain * error_y

        return vx, vy
```

---

## 🎯 11.9 Practical Project: Vision-Guided Robot Navigation

### Project Overview
Create a vision system that enables a humanoid robot to navigate through a simple environment while avoiding obstacles and following visual markers.

### Implementation Steps

#### **Step 1: Environment Setup**
```python
# vision_navigation_robot.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import cv2
import numpy as np

class VisionNavigationRobot(Node):
    def __init__(self):
        super().__init__('vision_navigation_robot')

        # Vision system
        self.vision_processor = VisionProcessor()

        # Control system
        self.controller = NavigationController()

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.vision_callback, 10
        )

        # Navigation parameters
        self.safe_distance = 1.0  # meters
        self.target_color_lower = np.array([40, 50, 50])  # Green
        self.target_color_upper = np.array([80, 255, 255])

    def vision_callback(self, msg):
        """Process camera images for navigation"""
        # Convert ROS 2 image to OpenCV
        cv_image = self.vision_processor.convert_ros_to_cv2(msg)

        # Detect target and obstacles
        target_info = self.detect_target(cv_image)
        obstacle_info = self.detect_obstacles(cv_image)

        # Compute navigation command
        cmd = self.controller.compute_navigation_command(
            target_info, obstacle_info
        )

        # Publish command
        self.cmd_vel_pub.publish(cmd)

    def detect_target(self, image):
        """Detect visual navigation target"""
        # Color-based target detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.target_color_lower, self.target_color_upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour
            target_contour = max(contours, key=cv2.contourArea)

            # Calculate center
            M = cv2.moments(target_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                area = cv2.contourArea(target_contour)

                return {'center': (cx, cy), 'area': area, 'detected': True}

        return {'detected': False}

    def detect_obstacles(self, image):
        """Detect obstacles using edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours of potential obstacles
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                obstacles.append({'bbox': (x, y, w, h), 'area': area})

        return obstacles
```

#### **Step 2: Navigation Controller**
```python
class NavigationController:
    def __init__(self):
        self.max_linear_velocity = 0.5  # m/s
        self.max_angular_velocity = 1.0  # rad/s

    def compute_navigation_command(self, target_info, obstacle_info):
        """Compute velocity command for navigation"""
        cmd = Twist()

        if not target_info['detected']:
            # Search for target
            cmd.angular.z = 0.5
            cmd.linear.x = 0.0
        else:
            # Move toward target
            target_center = target_info['center']
            target_area = target_info['area']

            # Calculate deviation from center (assuming 640x480 image)
            image_center_x = 320
            deviation = target_center[0] - image_center_x

            # Proportional control for angular velocity
            cmd.angular.z = -0.01 * deviation

            # Adjust linear velocity based on target size (distance)
            if target_area > 10000:  # Close to target
                cmd.linear.x = 0.1
            else:  # Far from target
                cmd.linear.x = 0.3

            # Obstacle avoidance
            obstacle_detected = False
            for obstacle in obstacle_info:
                bbox = obstacle['bbox']
                # Check if obstacle is in front
                if bbox[0] < 480 and bbox[0] + bbox[2] > 160:  # In center third
                    obstacle_detected = True
                    break

            if obstacle_detected:
                # Stop or avoid obstacle
                cmd.linear.x = 0.0
                cmd.angular.z = 0.8  # Turn to avoid

        # Limit velocities
        cmd.linear.x = np.clip(cmd.linear.x, -self.max_linear_velocity, self.max_linear_velocity)
        cmd.angular.z = np.clip(cmd.angular.z, -self.max_angular_velocity, self.max_angular_velocity)

        return cmd
```

---

## 🎓 **11.10 Advanced Research Topics**

### State-of-the-Art Computer Vision in Robotics

#### **Neural Radiance Fields (NeRF) for 3D Reconstruction**
Neural Radiance Fields represent a cutting-edge approach to 3D scene representation that has revolutionized robotics perception:

```python
import torch
import torch.nn as nn
import numpy as np

class NeRF(nn.Module):
    """Neural Radiance Field for novel view synthesis"""

    def __init__(self, hidden_dim=256, num_layers=8):
        super().__init__()

        # Positional encoding
        self.L_pos = 10  # Position encoding frequency
        self.L_dir = 4   # Direction encoding frequency

        # Network layers
        layers = []
        layers.append(nn.Linear(3 * 2 * self.L_pos + 3, hidden_dim))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Density prediction
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())

        # RGB and density output
        self.rgb_head = nn.Linear(hidden_dim + 3 * 2 * self.L_dir + 3, 3)
        self.density_head = nn.Linear(hidden_dim, 1)

        self.network = nn.Sequential(*layers)

    def positional_encoding(self, x, L):
        """Apply positional encoding to input coordinates"""
        encodings = [x]
        for l in range(L):
            encodings.append(torch.sin(2**l * x))
            encodings.append(torch.cos(2**l * x))
        return torch.cat(encodings, dim=-1)

    def forward(self, rays_o, rays_d, near=2.0, far=6.0, n_samples=64):
        # Sample points along rays
        t = torch.linspace(near, far, n_samples)
        t = t.expand(rays_o.shape[0], n_samples)

        # Calculate sample positions
        rays_o = rays_o[..., None, :].expand(-1, n_samples, -1)
        rays_d = rays_d[..., None, :].expand(-1, n_samples, -1)

        points = rays_o + t[..., None] * rays_d

        # Positional encoding
        points_enc = self.positional_encoding(points, self.L_pos)
        dirs_enc = self.positional_encoding(rays_d, self.L_dir)

        # Network forward pass
        h = self.network(points_enc)
        density = torch.relu(self.density_head(h))

        # Combine with direction for RGB
        h_rgb = torch.cat([h, dirs_enc.expand(-1, n_samples, -1)], dim=-1)
        rgb = torch.sigmoid(self.rgb_head(h_rgb))

        return rgb, density.squeeze(-1)

class RobotNeRFReconstruction:
    """NeRF-based 3D reconstruction for robotics"""

    def __init__(self, camera_intrinsics, image_size=(640, 480)):
        self.nerf = NeRF()
        self.camera_intrinsics = camera_intrinsics
        self.image_size = image_size
        self.optimizer = torch.optim.Adam(self.nerf.parameters(), lr=5e-4)

    def generate_rays(self, pose):
        """Generate camera rays from pose"""
        H, W = self.image_size
        i, j = torch.meshgrid(
            torch.arange(W), torch.arange(H),
            indexing='xy'
        )

        # Camera coordinates
        focal_length = self.camera_intrinsics[0, 0]
        dirs = torch.stack([
            (i - W * 0.5) / focal_length,
            (j - H * 0.5) / focal_length,
            torch.ones_like(i)
        ], dim=-1)

        # Transform to world coordinates
        rotation = pose[:3, :3]
        translation = pose[:3, 3:]

        rays_d = (rotation @ dirs.unsqueeze(-1)).squeeze(-1)
        rays_o = translation.expand_as(rays_d)

        return rays_o, rays_d

    def train_step(self, images, poses):
        """Single training step"""
        total_loss = 0

        for image, pose in zip(images, poses):
            rays_o, rays_d = self.generate_rays(pose)

            # Forward pass
            rgb, density = self.nerf(rays_o, rays_d)

            # Volume rendering
            weights = torch.softmax(density, dim=-1)
            rendered_rgb = torch.sum(weights[..., None] * rgb, dim=1)

            # Compute loss
            loss = torch.nn.functional.mse_loss(rendered_rgb, image)
            total_loss += loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

#### **Vision Transformers for Robotic Perception**
Vision Transformers (ViT) have shown remarkable performance in robotic perception tasks:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class VisionTransformer(nn.Module):
    """Vision Transformer for robot perception"""

    def __init__(self, img_size=224, patch_size=16,
                 embed_dim=768, num_heads=12, num_layers=12,
                 num_classes=1000):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)

        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer encoding
        x = x.transpose(0, 1)  # [seq_len, B, embed_dim]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [B, seq_len, embed_dim]

        # Classification
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits, x[:, 1:]  # Return patch embeddings too

class RobotVisionTransformer:
    """ViT-based perception for robotics"""

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained ViT
        self.vit = VisionTransformer(num_classes=1000)
        if model_path:
            self.vit.load_state_dict(torch.load(model_path))
        self.vit.to(self.device)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Attention visualization
        self.attention_weights = None

    def extract_features(self, image):
        """Extract patch embeddings for downstream tasks"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits, patch_embeddings = self.vit(img_tensor)

        return patch_embeddings.cpu().numpy().squeeze(0)

    def segment_image(self, image, num_classes=3):
        """Semantic segmentation using patch embeddings"""
        patch_embeddings = self.extract_features(image)

        # K-means clustering on patch embeddings
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_classes, random_state=42)
        cluster_labels = kmeans.fit_predict(patch_embeddings)

        # Reshape to image grid
        patch_size = 16
        h_patches = image.shape[0] // patch_size
        w_patches = image.shape[1] // patch_size

        segmentation_map = cluster_labels.reshape(h_patches, w_patches)
        segmentation_map = cv2.resize(
            segmentation_map.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        return segmentation_map
```

#### **Self-Supervised Learning for Robotic Vision**
Self-supervised learning enables robots to learn visual representations without human annotations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ContrastiveLearning(nn.Module):
    """Contrastive learning for self-supervised visual representation"""

    def __init__(self, temperature=0.07, feature_dim=128):
        super().__init__()

        # Backbone network (ResNet-50)
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()  # Remove classification head

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

        self.temperature = temperature

    def augment(self, x):
        """Data augmentation for contrastive learning"""
        # Random crop and resize
        x = transforms.RandomResizedCrop(224)(x)

        # Random horizontal flip
        x = transforms.RandomHorizontalFlip(p=0.5)(x)

        # Color jittering
        x = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                 saturation=0.4, hue=0.1)(x)

        # Gaussian blur
        x = transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))(x)

        return x

    def forward(self, x):
        # Generate two augmentations
        x1 = self.augment(x)
        x2 = self.augment(x)

        # Extract features
        feat1 = self.backbone(x1)
        feat2 = self.backbone(x2)

        # Project to contrastive space
        z1 = F.normalize(self.projection_head(feat1), dim=1)
        z2 = F.normalize(self.projection_head(feat2), dim=1)

        return z1, z2

    def contrastive_loss(self, z1, z2):
        """InfoNCE contrastive loss"""
        batch_size = z1.size(0)

        # Similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # Positive pairs
        pos_sim = torch.diag(sim_matrix)

        # Negative pairs
        neg_sim = sim_matrix - torch.eye(batch_size, device=sim_matrix.device)

        # InfoNCE loss
        loss = -pos_sim + torch.logsumexp(torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1), dim=1)
        loss = loss.mean()

        return loss

class RobotSelfSupervisedVision:
    """Self-supervised vision learning for robotics"""

    def __init__(self, data_loader=None):
        self.model = ContrastiveLearning()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=1e-4
        )

        self.data_loader = data_loader

    def train(self, num_epochs=100):
        """Train self-supervised model"""
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0

            for batch_idx, images in enumerate(self.data_loader):
                images = images.to(self.device)

                # Forward pass
                z1, z2 = self.model(images)

                # Compute loss
                loss = self.model.contrastive_loss(z1, z2)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {epoch_loss / len(self.data_loader):.4f}')

    def extract_representations(self, image):
        """Extract learned visual representations"""
        self.model.eval()

        with torch.no_grad():
            # Extract backbone features
            feat = self.model.backbone(image.unsqueeze(0).to(self.device))

        return feat.cpu().numpy().squeeze(0)
```

### **Experimental Validation Guidelines**

#### **Benchmark Datasets for Robotic Vision**
- **KITTI**: Autonomous driving and visual odometry
- **ImageNet LVIS**: Object detection with many categories
- **COCO**: Object detection and segmentation
- **ScanNet**: Indoor 3D scene understanding
- **Matterport3D**: Large-scale indoor environments
- **OpenImages**: Large-scale object detection
- **Cityscapes**: Urban scene segmentation

#### **Evaluation Metrics**
```python
def evaluate_vision_model(predictions, ground_truth, metrics):
    """Comprehensive evaluation of vision models"""

    results = {}

    # Classification metrics
    if 'classification' in metrics:
        results['accuracy'] = accuracy_score(ground_truth, predictions)
        results['precision'] = precision_score(ground_truth, predictions, average='weighted')
        results['recall'] = recall_score(ground_truth, predictions, average='weighted')
        results['f1'] = f1_score(ground_truth, predictions, average='weighted')

    # Detection metrics
    if 'detection' in metrics:
        results['map'] = compute_map(predictions, ground_truth)
        results['ap50'] = compute_ap_at_iou(predictions, ground_truth, iou_threshold=0.5)
        results['ap75'] = compute_ap_at_iou(predictions, ground_truth, iou_threshold=0.75)

    # Segmentation metrics
    if 'segmentation' in metrics:
        results['iou'] = compute_iou(predictions, ground_truth)
        results['pixel_accuracy'] = compute_pixel_accuracy(predictions, ground_truth)
        results['dice_coefficient'] = compute_dice_coefficient(predictions, ground_truth)

    # Real-time performance
    if 'performance' in metrics:
        results['fps'] = compute_fps(predictions)
        results['latency'] = compute_latency(predictions)
        results['memory_usage'] = compute_memory_usage(predictions)

    return results
```

---

## ✅ 11.11 Chapter Summary and Key Takeaways

### Core Concepts Covered
1. **OpenCV Fundamentals**: Installation, configuration, and basic image processing
2. **Feature Detection**: SIFT, ORB, and other feature extraction techniques
3. **Object Detection**: Template matching, color-based detection, and deep learning approaches
4. **3D Vision**: Stereo vision, depth estimation, and point cloud generation
5. **Real-time Processing**: Multi-threading and GPU acceleration
6. **ROS 2 Integration**: Vision nodes and visual servoing
7. **Practical Applications**: Navigation, human detection, and gesture recognition

### Key Skills Developed
- Setting up computer vision pipelines for humanoid robots
- Implementing real-time object detection and tracking
- Creating 3D perception systems
- Optimizing vision algorithms for performance
- Integrating vision with robot control systems

### Common Challenges and Solutions
- **Lighting Variations**: Use adaptive histogram equalization and robust feature descriptors
- **Real-time Constraints**: Implement multi-threading and GPU acceleration
- **Calibration Issues**: Regular camera calibration and robust parameter estimation
- **Computational Complexity**: Optimize algorithms and use appropriate data structures

### Best Practices
- **Modular Design**: Separate image processing, detection, and control components
- **Error Handling**: Implement robust error handling for camera failures
- **Performance Monitoring**: Track frame rates and processing times
- **Testing**: Test with various lighting conditions and environments

---

## 🚀 Next Steps

In the next chapter, we'll explore **Sensor Fusion** (Chapter 12), where we'll learn to combine multiple sensor modalities (vision, LiDAR, IMU) to create robust and reliable perception systems for humanoid robots.

### Preparation for Next Chapter
- Experiment with different computer vision techniques
- Practice optimizing vision algorithms for real-time performance
- Set up additional sensors (IMU, LiDAR) if available
- Review basic probability and statistics concepts for sensor fusion

**Remember**: Computer vision is a rapidly evolving field. Stay updated with the latest research and developments in areas like deep learning, neural networks, and edge AI computing for robotics! 🤖👁️