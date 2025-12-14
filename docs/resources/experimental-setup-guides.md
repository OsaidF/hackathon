---
title: "Experimental Setup Guides"
sidebar_label: "Experimental Setup Guides"
sidebar_position: 5
---

# Experimental Setup Guides for Robotics Research

This comprehensive guide provides detailed experimental setups for validating robotics research across different domains. Each setup includes hardware requirements, software configurations, data collection protocols, and evaluation methodologies.

## 📑 Table of Contents

1. [Computer Vision Experiments](#1-computer-vision-experiments)
2. [Sensor Fusion Validation](#2-sensor-fusion-validation)
3. [Deep Learning Benchmarking](#3-deep-learning-benchmarking)
4. [Human-Robot Interaction Studies](#4-human-robot-interaction-studies)
5. [Control System Validation](#5-control-system-validation)

---

## 1. Computer Vision Experiments

### **1.1 Object Detection Validation Setup**

#### **Hardware Requirements**
```
Camera System:
- Primary Camera: Intel RealSense D435i (RGB + Depth)
- Resolution: 1920x1080 @ 30 FPS
- Baseline: 55mm
- Depth Range: 0.2m - 10m

Computing Platform:
- CPU: Intel i7-10700K (8 cores, 3.8GHz)
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- RAM: 32GB DDR4
- Storage: 1TB NVMe SSD

Lighting Setup:
- LED Panel Lighting (5000K, adjustable intensity: 100-1000 lux)
- Diffusers for even illumination
- Light meter for calibration
```

#### **Software Configuration**
```bash
# Install required dependencies
sudo apt update
sudo apt install -y python3-pip cmake git

# Computer Vision Libraries
pip install opencv-python==4.8.1
pip install torch torchvision==2.0.1
pip install ultralytics
pip install transformers==4.30.0
pip install timm

# Data Processing
pip install numpy pandas matplotlib
pip install scikit-learn scipy
pip install jupyterlab

# ROS 2 Integration
source /opt/ros/humble/setup.bash
pip install rclpy cv-bridge sensor-msgs
```

#### **Data Collection Protocol**
```python
# experiment_setup.py
import cv2
import numpy as np
import json
import time
from datetime import datetime

class ComputerVisionExperiment:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        self.experiment_data = {
            'start_time': datetime.now().isoformat(),
            'frames_captured': 0,
            'detection_results': [],
            'performance_metrics': {}
        }

    def collect_dataset(self, duration_minutes=30, save_dir='./dataset'):
        """Collect controlled dataset for evaluation"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration_minutes * 60:
            ret, frame = self.camera.read()
            if not ret:
                continue

            timestamp = datetime.now().isoformat()
            filename = f"frame_{frame_count:06d}_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)

            cv2.imwrite(filepath, frame)
            frame_count += 1

            # Log frame metadata
            self.experiment_data['frames_captured'] = frame_count

            if frame_count % 100 == 0:
                print(f"Captured {frame_count} frames")

        print(f"Dataset collection complete: {frame_count} frames captured")
        return frame_count

    def evaluate_detection_model(self, model_path, test_data_dir):
        """Evaluate object detection model"""
        from ultralytics import YOLO

        # Load model
        model = YOLO(model_path)

        # Evaluation metrics storage
        results = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'inference_time': [],
            'confidence_scores': []
        }

        # Process test images
        import glob
        test_images = glob.glob(f"{test_data_dir}/*.jpg")

        for image_path in test_images[:100]:  # Limit for faster evaluation
            # Measure inference time
            start_time = time.time()
            detections = model(image_path)
            inference_time = time.time() - start_time

            # Store results
            results['inference_time'].append(inference_time)

            # Extract metrics (simplified)
            if len(detections) > 0 and len(detections[0].boxes) > 0:
                confidence = detections[0].boxes.conf.cpu().numpy().mean()
                results['confidence_scores'].append(confidence)

        # Calculate statistics
        self.experiment_data['performance_metrics'] = {
            'avg_inference_time': np.mean(results['inference_time']),
            'fps': 1 / np.mean(results['inference_time']),
            'avg_confidence': np.mean(results['confidence_scores']),
            'std_confidence': np.std(results['confidence_scores'])
        }

        return self.experiment_data['performance_metrics']

# Usage example
experiment = ComputerVisionExperiment()
experiment.collect_dataset(duration_minutes=5)
metrics = experiment.evaluate_detection_model('yolov8n.pt', './dataset')
print("Performance Metrics:", metrics)
```

#### **Evaluation Metrics**
```python
# evaluation_metrics.py
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

class DetectionEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.detection_confidences = []
        self.ground_truth_labels = []

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Union
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0

    def evaluate_detections(self, predictions, ground_truth):
        """Evaluate detection results"""
        matched_gt = set()

        for pred_box, pred_conf, pred_class in predictions:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, (gt_box, gt_class) in enumerate(ground_truth):
                if gt_idx in matched_gt or pred_class != gt_class:
                    continue

                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                self.true_positives += 1
                matched_gt.add(best_gt_idx)
                self.detection_confidences.append(pred_conf)
                self.ground_truth_labels.append(1)
            else:
                self.false_positives += 1
                self.detection_confidences.append(pred_conf)
                self.ground_truth_labels.append(0)

        # Count false negatives
        self.false_negatives = len(ground_truth) - len(matched_gt)

    def get_metrics(self):
        """Calculate evaluation metrics"""
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'average_precision': average_precision_score(self.ground_truth_labels, self.detection_confidences) if len(self.detection_confidences) > 0 else 0
        }
```

---

## 2. Sensor Fusion Validation

### **2.1 Visual-Inertial Odometry Setup**

#### **Hardware Configuration**
```
Sensor Suite:
- Visual Sensor: Intel RealSense D435i
  - RGB: 1920x1080 @ 30Hz
  - Depth: 1280x720 @ 30Hz
  - IMU: 6-axis (Accel + Gyro)
- External IMU: Xsens MTi-670G
  - Accelerometer: ±200g
  - Gyroscope: ±2000°/s
  - Magnetometer: ±8 Gauss
- Ground Truth: Vicon motion capture system
  - Accuracy: <0.1mm position, <0.1° orientation

Data Acquisition:
- Synchronization: Hardware trigger via Arduino
- Storage: High-speed SD card (100MB/s write)
- Power: UPS for uninterrupted operation
```

#### **Software Stack**
```python
# vio_experiment.py
import numpy as np
import cv2
import pyrealsense2 as rs
from datetime import datetime
import json
import threading
import queue

class VIOExperiment:
    def __init__(self):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure streams
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.accel, 200)
        self.config.enable_stream(rs.stream.gyro, 200)

        # Data storage
        self.data_queue = queue.Queue()
        self.recording = False

    def start_recording(self, duration_seconds=60):
        """Start synchronized data recording"""
        profile = self.pipeline.start(self.config)

        # Get device info for calibration
        device = profile.get_device()
        self.imu_sensor = device.first_motion_sensor()
        self.depth_sensor = device.first_depth_sensor()

        # Store calibration data
        calibration_data = {
            'accel_intrinsics': self.imu_sensor.get_motion_intrinsics(),
            'depth_intrinsics': self.depth_sensor.get_depth_intrinsics(),
            'start_time': datetime.now().isoformat()
        }

        self.recording = True
        start_time = time.time()

        # Start data collection thread
        collection_thread = threading.Thread(target=self._collect_data)
        collection_thread.start()

        # Record for specified duration
        while time.time() - start_time < duration_seconds:
            time.sleep(0.1)

        self.recording = False
        collection_thread.join()

        # Save data
        self._save_dataset('vio_dataset')

        return calibration_data

    def _collect_data(self):
        """Collect synchronized sensor data"""
        while self.recording:
            frames = self.pipeline.wait_for_frames()

            timestamp = frames.get_timestamp()

            # Process color frame
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # Process depth frame
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())

            # Process IMU data
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            data_point = {
                'timestamp': timestamp,
                'color_image': color_image,
                'depth_image': depth_image,
                'acceleration': accel_frame.get_motion_data() if accel_frame else None,
                'gyroscope': gyro_frame.get_motion_data() if gyro_frame else None
            }

            self.data_queue.put(data_point)

    def evaluate_vio_algorithm(self, algorithm, test_data):
        """Evaluate VIO algorithm performance"""
        estimated_trajectory = []
        ground_truth_trajectory = []

        # Process each frame
        for data_point in test_data:
            # Run VIO algorithm
            pose_estimate = algorithm.process_frame(data_point)
            estimated_trajectory.append(pose_estimate)

            # Get ground truth (if available)
            if 'ground_truth' in data_point:
                ground_truth_trajectory.append(data_point['ground_truth'])

        # Calculate evaluation metrics
        if len(ground_truth_trajectory) > 0:
            metrics = self._calculate_trajectory_metrics(
                estimated_trajectory,
                ground_truth_trajectory
            )
        else:
            metrics = {'trajectory_length': len(estimated_trajectory)}

        return metrics

    def _calculate_trajectory_metrics(self, estimated, ground_truth):
        """Calculate trajectory evaluation metrics"""
        estimated = np.array(estimated)
        ground_truth = np.array(ground_truth)

        # Absolute Trajectory Error (ATE)
        ate = np.sqrt(np.mean(np.sum((estimated - ground_truth)**2, axis=1)))

        # Relative Pose Error (RPE)
        rpe_translations = []
        for i in range(1, len(estimated)):
            est_delta = estimated[i] - estimated[i-1]
            gt_delta = ground_truth[i] - ground_truth[i-1]
            error = np.linalg.norm(est_delta - gt_delta)
            rpe_translations.append(error)

        rpe = np.mean(rpe_translations)

        return {
            'absolute_trajectory_error': ate,
            'relative_pose_error': rpe,
            'trajectory_length': len(estimated)
        }
```

---

## 3. Deep Learning Benchmarking

### **3.1 Neural Network Performance Validation**

#### **Hardware Benchmark Suite**
```python
# neural_network_benchmark.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import GPUtil
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworkBenchmark:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}

    def benchmark_model(self, model_class, model_params, input_shape, batch_size=32):
        """Comprehensive model benchmarking"""

        # Initialize model
        model = model_class(**model_params).to(self.device)

        # Generate synthetic data
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)

        # Benchmark metrics
        metrics = {
            'forward_pass_time': [],
            'backward_pass_time': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'model_size': 0,
            'parameter_count': 0
        }

        # Calculate model size
        metrics['model_size'] = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB
        metrics['parameter_count'] = sum(p.numel() for p in model.parameters())

        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)

        # Benchmark forward pass
        model.eval()
        with torch.no_grad():
            for _ in range(100):
                # Measure memory before
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                start_time = time.time()
                output = model(dummy_input)
                forward_time = time.time() - start_time

                metrics['forward_pass_time'].append(forward_time)

                # Record memory usage
                if torch.cuda.is_available():
                    metrics['memory_usage'].append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
                    metrics['gpu_utilization'].append(GPUtil.getGPUs()[0].load * 100)

        # Benchmark backward pass
        model.train()
        optimizer = optim.Adam(model.parameters())
        target = torch.randn(batch_size, 10).to(self.device)  # Dummy target
        criterion = nn.MSELoss()

        for _ in range(50):
            optimizer.zero_grad()

            start_time = time.time()
            output = model(dummy_input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            backward_time = time.time() - start_time

            metrics['backward_pass_time'].append(backward_time)

        # Calculate statistics
        results = {
            'avg_forward_time': np.mean(metrics['forward_pass_time']),
            'std_forward_time': np.std(metrics['forward_pass_time']),
            'avg_backward_time': np.mean(metrics['backward_pass_time']),
            'std_backward_time': np.std(metrics['backward_pass_time']),
            'avg_memory_usage': np.mean(metrics['memory_usage']),
            'peak_memory_usage': np.max(metrics['memory_usage']),
            'avg_gpu_utilization': np.mean(metrics['gpu_utilization']),
            'fps': 1 / np.mean(metrics['forward_pass_time']),
            'model_size_mb': metrics['model_size'],
            'parameter_count': metrics['parameter_count']
        }

        return results

# Example model for benchmarking
class BenchmarkCNN(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=256):
        super().__init__()

        layers = []
        in_channels = 3

        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Run benchmark
benchmark = NeuralNetworkBenchmark()
results = benchmark.benchmark_model(
    BenchmarkCNN,
    {'num_layers': 4, 'hidden_dim': 256},
    (3, 224, 224),
    batch_size=32
)

print("Benchmark Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

---

## 4. Human-Robot Interaction Studies

### **4.1 User Experience Evaluation Framework**

#### **Study Design Template**
```python
# hri_study_framework.py
import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class HRIStudyFramework:
    def __init__(self, study_name):
        self.study_name = study_name
        self.participants = []
        self.metrics = {}
        self.start_time = None

    def setup_study(self, participant_count=30):
        """Setup HRI study parameters"""
        self.study_config = {
            'participant_demographics': {
                'age_groups': ['18-25', '26-35', '36-45', '46-55', '55+'],
                'technical_background': ['None', 'Basic', 'Intermediate', 'Advanced'],
                'robotics_experience': ['None', 'Minimal', 'Moderate', 'Extensive']
            },
            'evaluation_scales': {
                'system_usability_scale': range(1, 6),
                'likert_scale': range(1, 8),
                'trust_scale': range(1, 8),
                'anthropomorphism_scale': range(1, 6)
            },
            'task_scenarios': [
                'Object Manipulation',
                'Navigation Assistance',
                'Social Interaction',
                'Emergency Response'
            ]
        }

        print(f"Study '{study_name}' configured for {participant_count} participants")
        return self.study_config

    def collect_demographics(self, participant_id):
        """Collect participant demographic information"""
        demographics = {
            'participant_id': participant_id,
            'age': input("Participant age: "),
            'gender': input("Participant gender: "),
            'education': input("Highest education level: "),
            'technical_background': input("Technical background (None/Basic/Intermediate/Advanced): "),
            'robotics_experience': input("Robotics experience (None/Minimal/Moderate/Extensive): "),
            'timestamp': datetime.now().isoformat()
        }

        return demographics

    def administer_sus_questionnaire(self):
        """System Usability Scale questionnaire"""
        sus_questions = [
            "I think that I would like to use this system frequently.",
            "I found the system unnecessarily complex.",
            "I thought the system was easy to use.",
            "I think that I would need the support of a technical person to be able to use this system.",
            "I found the various functions in this system were well integrated.",
            "I thought there was too much inconsistency in this system.",
            "I would imagine that most people would learn to use this system very quickly.",
            "I found the system very cumbersome to use.",
            "I felt very confident using the system.",
            "I needed to learn a lot of things before I could get going with this system."
        ]

        responses = []
        print("\n=== System Usability Scale ===")
        print("Rate each statement on a scale of 1-5 (1=Strongly Disagree, 5=Strongly Agree)")

        for i, question in enumerate(sus_questions):
            while True:
                try:
                    response = int(input(f"{i+1}. {question} (1-5): "))
                    if 1 <= response <= 5:
                        responses.append(response)
                        break
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")

        # Calculate SUS score
        odd_positions = responses[::2]  # 1-based: 1, 3, 5, 7, 9
        even_positions = responses[1::2]  # 1-based: 2, 4, 6, 8, 10

        # Convert to 0-4 scale
        odd_converted = [x - 1 for x in odd_positions]
        even_converted = [5 - x for x in even_positions]

        sus_score = sum(odd_converted + even_converted) * 2.5

        return {
            'individual_responses': responses,
            'sus_score': sus_score,
            'interpretation': self._interpret_sus_score(sus_score)
        }

    def _interpret_sus_score(self, score):
        """Interpret SUS score"""
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "OK"
        elif score >= 30:
            return "Poor"
        else:
            return "Awful"

    def measure_task_performance(self, task_name, participant_id):
        """Measure participant task performance metrics"""
        print(f"\n=== Task Performance: {task_name} ===")
        print("Press Enter to start the task, and Enter again when completed.")

        input("Press Enter to start...")
        start_time = time.time()

        # Here you would integrate with your actual robot system
        # For demonstration, we'll just measure the time
        task_data = {
            'task_name': task_name,
            'participant_id': participant_id,
            'start_time': datetime.now().isoformat(),
            'interactions': [],  # Would be filled by actual robot interaction data
            'errors': [],        # Would track any errors or failures
        }

        input("Press Enter when task is completed...")
        end_time = time.time()

        task_data.update({
            'end_time': datetime.now().isoformat(),
            'duration_seconds': end_time - start_time,
            'success': input("Was the task completed successfully? (y/n): ").lower() == 'y',
            'difficulty_rating': int(input("Rate task difficulty (1-7, 1=Very Easy, 7=Very Difficult): ")),
            'satisfaction_rating': int(input("Rate satisfaction with result (1-7, 1=Very Dissatisfied, 7=Very Satisfied): "))
        })

        return task_data

    def analyze_study_data(self):
        """Analyze collected study data"""
        if not self.participants:
            print("No participant data available for analysis.")
            return

        # Demographics analysis
        ages = [p['demographics']['age'] for p in self.participants if 'demographics' in p]
        print(f"Participant Age Statistics:")
        print(f"  Mean: {np.mean([float(age) for age in ages if age.replace('.', '', 1).isdigit()]):.1f}")
        print(f"  Count: {len(ages)}")

        # SUS score analysis
        sus_scores = [p['sus_results']['sus_score'] for p in self.participants if 'sus_results' in p]
        if sus_scores:
            print(f"\nSUS Score Statistics:")
            print(f"  Mean: {np.mean(sus_scores):.2f}")
            print(f"  Std: {np.std(sus_scores):.2f}")
            print(f"  Min: {np.min(sus_scores):.2f}")
            print(f"  Max: {np.max(sus_scores):.2f}")

        # Task performance analysis
        task_times = []
        task_success_rates = []

        for participant in self.participants:
            if 'tasks' in participant:
                for task in participant['tasks']:
                    task_times.append(task['duration_seconds'])
                    task_success_rates.append(1 if task['success'] else 0)

        if task_times:
            print(f"\nTask Performance Statistics:")
            print(f"  Average Duration: {np.mean(task_times):.2f} seconds")
            print(f"  Success Rate: {np.mean(task_success_rates)*100:.1f}%")

    def save_study_data(self, filename):
        """Save study data to JSON file"""
        study_data = {
            'study_name': self.study_name,
            'study_config': getattr(self, 'study_config', {}),
            'participants': self.participants,
            'analysis_date': datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(study_data, f, indent=2)

        print(f"Study data saved to {filename}")

# Usage example
# study = HRIStudyFramework("Robot_Assistant_Evaluation")
# study.setup_study(participant_count=10)
#
# # Simulate participant data collection
# participant_id = "P001"
# demographics = study.collect_demographics(participant_id)
# sus_results = study.administer_sus_questionnaire()
#
# participant_data = {
#     'participant_id': participant_id,
#     'demographics': demographics,
#     'sus_results': sus_results,
#     'tasks': []
# }
#
# # Add task performance data
# for task in ['Object Manipulation', 'Navigation']:
#     task_data = study.measure_task_performance(task, participant_id)
#     participant_data['tasks'].append(task_data)
#
# study.participants.append(participant_data)
# study.analyze_study_data()
# study.save_study_data('hri_study_results.json')
```

---

## 5. Control System Validation

### **5.1 PID Controller Performance Testing**

#### **Control System Benchmark**
```python
# control_system_validation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ctrl
import time

class ControlSystemValidator:
    def __init__(self):
        self.results = {}

    def pid_controller_benchmark(self, plant, pid_params, reference_signal, duration=10.0, dt=0.01):
        """Benchmark PID controller performance"""

        Kp, Ki, Kd = pid_params

        # Time vector
        t = np.arange(0, duration, dt)

        # System state variables
        integral = 0
        prev_error = 0

        # Storage for results
        output = np.zeros_like(t)
        error_signal = np.zeros_like(t)
        control_signal = np.zeros_like(t)

        # Simulation loop
        for i, time_point in enumerate(t):
            # Current reference and output
            reference = reference_signal[i] if i < len(reference_signal) else reference_signal[-1]
            current_output = output[i-1] if i > 0 else 0

            # Calculate error
            error = reference - current_output
            error_signal[i] = error

            # PID control law
            integral += error * dt
            derivative = (error - prev_error) / dt if i > 0 else 0

            control = Kp * error + Ki * integral + Kd * derivative
            control_signal[i] = control

            # Apply control to plant (simplified - in real system this would be the actual plant dynamics)
            # For demonstration, using simple first-order system
            tau = 1.0  # Time constant
            if i == 0:
                output[i] = current_output + (control - current_output) * dt / tau
            else:
                output[i] = output[i-1] + (control - output[i-1]) * dt / tau

            prev_error = error

        # Calculate performance metrics
        metrics = self._calculate_control_metrics(reference_signal[:len(output)], output, error_signal, control_signal, t)

        return {
            'time': t,
            'output': output,
            'reference': reference_signal[:len(output)],
            'error': error_signal,
            'control': control_signal,
            'metrics': metrics
        }

    def _calculate_control_metrics(self, reference, output, error, control, time):
        """Calculate control performance metrics"""

        # Steady-state error
        steady_state_error = np.mean(error[-int(len(error)*0.1):])

        # Rise time (10% to 90% of final value)
        final_value = reference[-1]
        if final_value != 0:
            rise_start_idx = np.where(output >= 0.1 * final_value)[0]
            rise_end_idx = np.where(output >= 0.9 * final_value)[0]

            if len(rise_start_idx) > 0 and len(rise_end_idx) > 0:
                rise_time = time[rise_end_idx[0]] - time[rise_start_idx[0]]
            else:
                rise_time = np.nan
        else:
            rise_time = np.nan

        # Settling time (2% criterion)
        settling_band = 0.02 * final_value
        settling_indices = np.where(np.abs(output - final_value) <= settling_band)[0]

        if len(settling_indices) > 0:
            settling_time = time[settling_indices[0]]
        else:
            settling_time = np.nan

        # Overshoot
        max_output = np.max(output)
        overshoot = ((max_output - final_value) / final_value * 100) if final_value != 0 else 0

        # Control effort (integral of absolute control signal)
        control_effort = np.trapz(np.abs(control), time)

        # RMSE
        rmse = np.sqrt(np.mean(error**2))

        return {
            'steady_state_error': steady_state_error,
            'rise_time': rise_time,
            'settling_time': settling_time,
            'overshoot_percent': overshoot,
            'control_effort': control_effort,
            'rmse': rmse,
            'final_value': output[-1]
        }

# Usage example
validator = ControlSystemValidator()

# Define reference signals
step_signal = np.concatenate([np.zeros(50), np.ones(150)])
sinusoid_signal = np.sin(np.linspace(0, 4*np.pi, 200))

# Test different PID parameters
pid_configs = [
    (1.0, 0.1, 0.05),  # Conservative
    (2.0, 0.5, 0.1),   # Moderate
    (5.0, 2.0, 0.5),   # Aggressive
]

results = {}
for i, pid_params in enumerate(pid_configs):
    result = validator.pid_controller_benchmark(
        plant=None,  # Using simplified plant in benchmark
        pid_params=pid_params,
        reference_signal=step_signal
    )
    results[f'PID_Config_{i+1}'] = result

    print(f"PID Configuration {i+1} (Kp={pid_params[0]}, Ki={pid_params[1]}, Kd={pid_params[2]}):")
    print(f"  Rise Time: {result['metrics']['rise_time']:.3f}s")
    print(f"  Settling Time: {result['metrics']['settling_time']:.3f}s")
    print(f"  Overshoot: {result['metrics']['overshoot_percent']:.1f}%")
    print(f"  RMSE: {result['metrics']['rmse']:.4f}")
    print()
```

---

## 6. Statistical Analysis Methods

### **6.1 Experimental Statistics Framework**

```python
# statistical_analysis.py
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel, f_oneway, chi2_contingency
from statsmodels.stats.power import ttest_power
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class ExperimentalStatistics:
    def __init__(self):
        self.results = {}

    def sample_size_calculation(self, effect_size, alpha=0.05, power=0.8, test_type='two_sample'):
        """Calculate required sample size for statistical tests"""
        if test_type == 'two_sample':
            sample_size = ttest_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )
        else:
            # For other test types, would need different calculations
            sample_size = None

        return {
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'required_sample_size': sample_size,
            'test_type': test_type
        }

    def normality_test(self, data, alpha=0.05):
        """Test for normality in data"""
        data = np.array(data)

        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))

        # Anderson-Darling test
        ad_stat, ad_critical_values, ad_significance_levels = stats.anderson(data, dist='norm')

        return {
            'sample_size': len(data),
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'shapiro_wilk': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > alpha
            },
            'kolmogorov_smirnov': {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > alpha
            },
            'anderson_darling': {
                'statistic': ad_stat,
                'critical_values': ad_critical_values,
                'significance_levels': ad_significance_levels
            }
        }

    def compare_two_groups(self, group1, group2, test_type='auto', alpha=0.05):
        """Compare two groups using appropriate statistical test"""
        group1 = np.array(group1)
        group2 = np.array(group2)

        # Check normality
        normal1 = stats.shapiro(group1)[1] > alpha
        normal2 = stats.shapiro(group2)[1] > alpha

        # Check variance equality
        _, levene_p = stats.levene(group1, group2)
        equal_var = levene_p > alpha

        # Select appropriate test
        if test_type == 'auto':
            if normal1 and normal2:
                if equal_var:
                    test_result = stats.ttest_ind(group1, group2, equal_var=True)
                    test_name = "Student's t-test"
                else:
                    test_result = stats.ttest_ind(group1, group2, equal_var=False)
                    test_name = "Welch's t-test"
            else:
                test_result = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                test_name = "Mann-Whitney U test"
        elif test_type == 't_test':
            test_result = stats.ttest_ind(group1, group2, equal_var=equal_var)
            test_name = "Student's t-test"
        elif test_type == 'mann_whitney':
            test_result = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U test"

        # Calculate effect size
        if normal1 and normal2:
            # Cohen's d
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                                 (len(group2) - 1) * np.var(group2, ddof=1)) /
                                (len(group1) + len(group2) - 2))
            effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
        else:
            # Rank-biserial correlation
            n1, n2 = len(group1), len(group2)
            U = test_result.statistic if hasattr(test_result, 'statistic') else test_result[0]
            effect_size = 1 - (2 * U) / (n1 * n2)

        return {
            'test_name': test_name,
            'group1_stats': {
                'n': len(group1),
                'mean': np.mean(group1),
                'std': np.std(group1, ddof=1),
                'normal': normal1
            },
            'group2_stats': {
                'n': len(group2),
                'mean': np.mean(group2),
                'std': np.std(group2, ddof=1),
                'normal': normal2
            },
            'test_statistic': test_result.statistic if hasattr(test_result, 'statistic') else test_result[0],
            'p_value': test_result.pvalue if hasattr(test_result, 'pvalue') else test_result[1],
            'effect_size': effect_size,
            'equal_variances': equal_var,
            'significant': test_result.pvalue if hasattr(test_result, 'pvalue') else test_result[1] < alpha
        }

    def anova_test(self, groups, alpha=0.05):
        """Perform one-way ANOVA test"""
        # Perform ANOVA
        f_stat, p_value = f_oneway(*groups)

        # Calculate effect size (eta-squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        # Between-group sum of squares
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)

        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean)**2)

        # Eta-squared
        eta_squared = ss_between / ss_total

        # Post-hoc test if significant
        post_hoc = None
        if p_value < alpha:
            # Prepare data for Tukey's test
            data_values = np.concatenate(groups)
            data_groups = []
            for i, group in enumerate(groups):
                data_groups.extend([f'Group_{i+1}'] * len(group))

            post_hoc = pairwise_tukeyhsd(data_values, data_groups)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < alpha,
            'group_stats': [
                {
                    'group': f'Group_{i+1}',
                    'n': len(group),
                    'mean': np.mean(group),
                    'std': np.std(group, ddof=1)
                }
                for i, group in enumerate(groups)
            ],
            'post_hoc': post_hoc
        }

    def correlation_analysis(self, x, y, method='pearson'):
        """Perform correlation analysis"""
        x, y = np.array(x), np.array(y)

        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(x, y)

        return {
            'correlation_coefficient': corr,
            'p_value': p_value,
            'method': method,
            'significant': p_value < 0.05,
            'sample_size': len(x)
        }

    def generate_report(self, results_data, output_file='statistical_report.html'):
        """Generate HTML report of statistical analysis"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Statistical Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .significant {{ color: red; font-weight: bold; }}
                .not-significant {{ color: green; }}
            </style>
        </head>
        <body>
            <h1>Statistical Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """

        for test_name, test_results in results_data.items():
            html_content += f"<h2>{test_name}</h2>"

            if 'group1_stats' in test_results:  # Two-group comparison
                html_content += f"""
                <table>
                    <tr><th>Metric</th><th>Group 1</th><th>Group 2</th></tr>
                    <tr><td>Sample Size</td><td>{test_results['group1_stats']['n']}</td><td>{test_results['group2_stats']['n']}</td></tr>
                    <tr><td>Mean</td><td>{test_results['group1_stats']['mean']:.4f}</td><td>{test_results['group2_stats']['mean']:.4f}</td></tr>
                    <tr><td>Std Dev</td><td>{test_results['group1_stats']['std']:.4f}</td><td>{test_results['group2_stats']['std']:.4f}</td></tr>
                </table>

                <h3>Test Results</h3>
                <table>
                    <tr><th>Test</th><th>Statistic</th><th>p-value</th><th>Effect Size</th><th>Significance</th></tr>
                    <tr>
                        <td>{test_results['test_name']}</td>
                        <td>{test_results['test_statistic']:.4f}</td>
                        <td>{test_results['p_value']:.4f}</td>
                        <td>{test_results['effect_size']:.4f}</td>
                        <td class="{'significant' if test_results['significant'] else 'not-significant'}">
                            {'Significant' if test_results['significant'] else 'Not Significant'}
                        </td>
                    </tr>
                </table>
                """

        html_content += """
        </body>
        </html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"Statistical report saved to {output_file}")

# Usage example
stats_analyzer = ExperimentalStatistics()

# Example: Compare two algorithm performances
algorithm1_performance = [85.2, 87.1, 83.9, 88.5, 86.3, 84.7, 87.9, 85.6]
algorithm2_performance = [82.1, 83.5, 81.9, 84.2, 82.8, 83.7, 82.4, 83.1]

comparison_result = stats_analyzer.compare_two_groups(
    algorithm1_performance,
    algorithm2_performance
)

print("Algorithm Comparison Results:")
print(f"Test: {comparison_result['test_name']}")
print(f"p-value: {comparison_result['p_value']:.4f}")
print(f"Effect size: {comparison_result['effect_size']:.4f}")
print(f"Significant: {comparison_result['significant']}")
```

---

## 📊 **Best Practices for Experimental Design**

### **Experimental Design Principles**

1. **Control Variables**: Identify and control all relevant variables except the independent variable
2. **Randomization**: Randomly assign participants/conditions to minimize bias
3. **Blinding**: Use single or double-blind procedures where applicable
4. **Replication**: Conduct multiple trials to ensure reliability
5. **Sample Size**: Use power analysis to determine appropriate sample sizes

### **Data Quality Assurance**

1. **Data Validation**: Implement data validation checks and outlier detection
2. **Missing Data Handling**: Establish protocols for handling missing or corrupted data
3. **Calibration**: Regular calibration of sensors and measurement equipment
4. **Documentation**: Comprehensive documentation of experimental procedures

### **Statistical Rigor**

1. **Pre-registration**: Register experimental protocols before data collection
2. **Multiple Comparisons**: Apply appropriate corrections for multiple testing
3. **Effect Size Reporting**: Report both statistical significance and effect sizes
4. **Confidence Intervals**: Include confidence intervals for key estimates

### **Reproducibility Standards**

1. **Code Availability**: Share analysis code and experimental scripts
2. **Data Sharing**: Provide access to anonymized datasets where appropriate
3. **Environment Documentation**: Document software versions and hardware specifications
4. **Step-by-step Procedures**: Provide detailed replication instructions

---

This comprehensive experimental setup guide provides researchers with standardized protocols for conducting rigorous and reproducible robotics research across different domains. The framework ensures scientific validity while maintaining practical feasibility for real-world experimental conditions.