---
title: "Chapter 15: Edge Deployment"
sidebar_label: "Chapter 15: Edge Deployment"
sidebar_position: 15
---

# Chapter 15: Edge Deployment

## Deploying AI Models on Embedded Systems

Welcome to Chapter 15! This final chapter of Quarter 3 focuses on the critical transition from simulation and development to real-world deployment on edge devices. You'll learn how to optimize and deploy AI models for humanoid robots on resource-constrained embedded systems while maintaining real-time performance.

## 🎯 Chapter Overview

### Learning Objectives
By the end of this chapter, you will be able to:
- Optimize AI models for embedded deployment on NVIDIA Jetson platforms
- Implement real-time inference pipelines for robotic perception and control
- Design cloud-edge hybrid architectures for scalable robot systems
- Apply performance tuning techniques for resource-constrained environments
- Deploy complete perception-action loops on edge devices
- Monitor and maintain edge AI systems in production

### Prerequisites
- **Chapter 11**: Computer Vision fundamentals
- **Chapter 13**: Perception Algorithms and deep learning
- **Chapter 14**: Isaac Sim simulation experience
- Basic understanding of containerization (Docker)
- Linux system administration skills

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|----------|-------------|-------|
| **Edge Device** | Jetson Nano 2GB | Jetson AGX Orin 64GB | Memory critical for model deployment |
| **Storage** | 32GB eMMC | 128GB NVMe SSD | Fast storage reduces loading times |
| **Camera** | USB Webcam | MIPI CSI Camera | Low latency critical |
| **Power** | 5V 4A | 19V 9A | Stable power under load |
| **Cooling** | Passive | Active cooling fan | Prevents thermal throttling |

## 🔧 Technical Foundations

### Edge AI Architecture

#### **System Components**

1. **Model Optimization Layer**
   - Quantization and pruning
   - TensorRT optimization
   - Model compression techniques

2. **Runtime Environment**
   - CUDA runtime
   - TensorRT inference engine
   - ONNX runtime support

3. **Application Layer**
   - ROS 2 nodes
   - Real-time control loops
   - Sensor data processing

4. **System Management**
   - Resource monitoring
   - Thermal management
   - Failover mechanisms

#### **Deployment Pipeline**

```python
# Edge AI Deployment Pipeline
class EdgeAIDeployment:
    def __init__(self):
        self.device_type = self.detect_device()
        self.memory_budget = self.get_memory_budget()
        self.compute_capability = self.get_compute_capability()

    def prepare_model(self, model_path):
        # Load and optimize model for edge
        model = self.load_model(model_path)
        optimized_model = self.optimize_for_edge(model)
        return optimized_model

    def deploy_pipeline(self, model, config):
        # Deploy complete inference pipeline
        self.setup_runtime()
        self.allocate_resources()
        self.start_inference_service(model, config)
        self.monitor_performance()
```

## 📦 Edge Deployment Platforms

### NVIDIA Jetson Family

#### **Platform Comparison**

| Feature | Nano | Xavier NX | AGX Orin |
|---------|------|-----------|----------|
| **GPU Cores** | 128 | 384 | 2048 |
| **Memory** | 2-4 GB | 8 GB | 32-64 GB |
| **TOPS AI** | 472 | 21 | 275 |
| **Power** | 5-10W | 10-20W | 15-60W |
| **Price** | $149 | $399 | $1999 |

#### **Platform Selection Guide**

```python
# Platform Selection based on requirements
def select_jetson_platform(requirements):
    models = requirements.get('models', [])
    resolution = requirements.get('resolution', '720p')
    fps_target = requirements.get('fps', 30)
    power_constraint = requirements.get('power_budget', 15)

    # Calculate requirements
    memory_needed = sum([model.memory_size for model in models])
    compute_needed = sum([model.flops * fps_target for model in models])

    if memory_needed < 2e9 and compute_needed < 1e11 and power_constraint < 10:
        return "Jetson Nano"
    elif memory_needed < 6e9 and compute_needed < 5e11 and power_constraint < 20:
        return "Jetson Xavier NX"
    else:
        return "Jetson AGX Orin"
```

### JetPack SDK Setup

#### **Installation Process**

```bash
# 1. Flash JetPack SDK
# Download NVIDIA SDK Manager
sudo apt install nvidia-jetpack

# 2. Configure development environment
sudo apt update
sudo apt install python3-pip python3-dev
sudo apt install cmake ninja-build

# 3. Install AI frameworks
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install tensorflow
pip3 install onnx onnxruntime-gpu

# 4. Install ROS 2 Humble
sudo apt install ros-humble-desktop
sudo apt install ros-humble-vision-opencv ros-humble-image-transport

# 5. Configure CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## 🚀 Model Optimization

### Quantization Techniques

#### **Post-Training Quantization (PTQ)**

```python
import torch
import torch.quantization

class QuantizedVisionModel:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def quantize_dynamic(self):
        # Dynamic quantization for inference
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        return quantized_model

    def quantize_static(self, calibration_loader):
        # Static quantization with calibration
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)

        # Calibrate with sample data
        with torch.no_grad():
            for data, _ in calibration_loader:
                self.model(data)

        quantized_model = torch.quantization.convert(self.model, inplace=False)
        return quantized_model
```

#### **TensorRT Optimization**

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTOptimizer:
    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING)

    def build_engine(self, onnx_path, max_batch_size=1):
        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        # Build optimized engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # Use FP16 precision

        engine = builder.build_engine(network, config)
        return engine

    def optimize_for_jetson(self, onnx_path, precision='fp16'):
        # Optimize for specific Jetson platform
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)

        # Parse model
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        # Configuration for Jetson
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 29  # 512MB for Jetson Nano

        if precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            # Add calibration data for INT8
        elif precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)

        # Enable DLA (Deep Learning Accelerator) if available
        if builder.num_dla_cores > 0:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = 0
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        engine = builder.build_engine(network, config)
        return engine
```

### Model Compression

#### **Pruning Techniques**

```python
import torch.nn.utils.prune as prune

class ModelPruner:
    def __init__(self, model):
        self.model = model

    def structured_pruning(self, pruning_ratio=0.2):
        # Remove entire neurons/channels
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')
        return self.model

    def unstructured_pruning(self, pruning_ratio=0.1):
        # Remove individual weights
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')
        return self.model

    def magnitude_based_pruning(self, sparsity=0.5):
        # Global magnitude-based pruning
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )

        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        return self.model
```

## 🏗️ Real-Time Inference

### Inference Pipeline Architecture

#### **Multi-Threaded Processing**

```python
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

class RealTimeInference:
    def __init__(self, model, input_shape, target_fps=30):
        self.model = model
        self.input_shape = input_shape
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        # Thread-safe queues
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)

        # Performance tracking
        self.inference_times = []
        self.fps_history = []

    def preprocessing_worker(self):
        # Continuous preprocessing thread
        while True:
            if not self.input_queue.empty():
                frame = self.input_queue.get()
                processed = self.preprocess(frame)
                self.output_queue.put(processed)
            time.sleep(0.001)  # Small delay to prevent CPU overload

    def inference_worker(self):
        # Dedicated inference thread
        while True:
            if not self.output_queue.empty():
                processed_input = self.output_queue.get()

                start_time = time.time()
                with torch.no_grad():
                    output = self.model(processed_input)
                inference_time = time.time() - start_time

                self.inference_times.append(inference_time)
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)

                # Process output
                self.postprocess_output(output)
            time.sleep(0.001)

    def start_pipeline(self):
        # Start worker threads
        self.preprocess_thread = threading.Thread(target=self.preprocessing_worker)
        self.inference_thread = threading.Thread(target=self.inference_worker)

        self.preprocess_thread.start()
        self.inference_thread.start()

    def process_frame(self, frame):
        # Main processing entry point
        self.input_queue.put(frame)

        # FPS calculation
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
        self.last_frame_time = current_time
```

#### **Batch Processing for Efficiency**

```python
class BatchInference:
    def __init__(self, model, max_batch_size=8, timeout_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms

        self.batch_queue = []
        self.batch_lock = threading.Lock()
        self.last_batch_time = time.time()

    def add_to_batch(self, frame, callback):
        # Add frame to current batch
        with self.batch_lock:
            self.batch_queue.append((frame, callback))

            # Process batch if conditions met
            if (len(self.batch_queue) >= self.max_batch_size or
                time.time() - self.last_batch_time > self.timeout_ms / 1000.0):
                self.process_batch()

    def process_batch(self):
        if not self.batch_queue:
            return

        # Extract frames and callbacks
        frames, callbacks = zip(*self.batch_queue)
        self.batch_queue.clear()

        # Preprocess batch
        batch_tensor = torch.stack([self.preprocess(frame) for frame in frames])

        # Batch inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)

        # Postprocess and callback
        for i, callback in enumerate(callbacks):
            output = outputs[i:i+1]  # Single output for this frame
            processed = self.postprocess(output)
            callback(processed)

        self.last_batch_time = time.time()
```

## 🌐 Cloud-Edge Hybrid Architecture

### Distributed Computing

#### **Edge-Cloud Communication**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from custom_interfaces.msg import InferenceRequest, InferenceResponse

class EdgeCloudNode(Node):
    def __init__(self):
        super().__init__('edge_cloud_node')

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.result_pub = self.create_publisher(
            InferenceResponse, '/inference/results', 10)

        # Cloud service client
        self.cloud_client = self.create_client(
            InferenceService, '/cloud_inference')

        # Local inference flag
        self.use_cloud_inference = False
        self.confidence_threshold = 0.8

    def image_callback(self, msg):
        # Decide local vs cloud inference
        inference_needed = self.determine_inference_needs(msg)

        if inference_needed == 'local':
            self.local_inference(msg)
        elif inference_needed == 'cloud':
            self.cloud_inference(msg)
        else:
            self.get_logger().info('Inference not needed')

    def determine_inference_needs(self, image_msg):
        # Determine where to process based on requirements
        image_size = image_msg.width * image_msg.height

        if image_size > 1920 * 1080:  # High resolution -> cloud
            return 'cloud'
        elif self.use_cloud_inference:  # Manual override
            return 'cloud'
        else:
            return 'local'

    def local_inference(self, msg):
        # Process locally on edge device
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.local_model.process(cv_image)

            # Check confidence and potentially fallback to cloud
            max_confidence = max([r.confidence for r in results])
            if max_confidence < self.confidence_threshold:
                self.cloud_inference(msg)
                return

            # Publish local results
            response = InferenceResponse()
            response.header = msg.header
            response.source = 'local_edge'
            response.results = results
            self.result_pub.publish(response)

        except Exception as e:
            self.get_logger().error(f'Local inference failed: {e}')
            # Fallback to cloud
            self.cloud_inference(msg)

    def cloud_inference(self, msg):
        # Send to cloud for processing
        request = InferenceRequest()
        request.image = msg
        request.priority = self.determine_priority()

        future = self.cloud_client.call_async(request)
        future.add_done_callback(self.cloud_response_callback)

    def cloud_response_callback(self, future):
        try:
            response = future.result()
            response.source = 'cloud_server'
            self.result_pub.publish(response)
        except Exception as e:
            self.get_logger().error(f'Cloud inference failed: {e}')
```

#### **Adaptive Load Balancing**

```python
class AdaptiveLoadBalancer:
    def __init__(self, local_model, cloud_client):
        self.local_model = local_model
        self.cloud_client = cloud_client

        # Performance metrics
        self.local_inference_times = deque(maxlen=100)
        self.cloud_inference_times = deque(maxlen=100)
        self.system_load = deque(maxlen=100)

        # Adaptive thresholds
        self.load_threshold = 0.8  # System load threshold
        self.latency_threshold = 0.1  # 100ms

    def route_inference(self, data, priority='normal'):
        current_load = self.get_system_load()
        local_latency = self.estimate_local_latency()
        cloud_latency = self.estimate_cloud_latency()

        # Decision logic
        if priority == 'high':
            # High priority always local for low latency
            return self.local_inference(data)
        elif current_load > self.load_threshold:
            # High system load -> use cloud
            return self.cloud_inference(data)
        elif local_latency > self.latency_threshold * 2:
            # Local too slow -> use cloud
            return self.cloud_inference(data)
        else:
            # Default to local
            return self.local_inference(data)

    def get_system_load(self):
        # Get current system load (CPU, GPU, memory)
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            gpu_utilization = self.get_gpu_utilization()

            return max(cpu_percent, memory_percent, gpu_utilization) / 100.0
        except:
            return 0.5  # Default moderate load

    def estimate_local_latency(self):
        if not self.local_inference_times:
            return 0.05  # Default 50ms

        return sum(self.local_inference_times) / len(self.local_inference_times)

    def update_metrics(self, source, inference_time):
        if source == 'local':
            self.local_inference_times.append(inference_time)
        elif source == 'cloud':
            self.cloud_inference_times.append(inference_time)

        self.system_load.append(self.get_system_load())
```

## 📊 Performance Monitoring

### Resource Management

#### **Real-Time Monitoring**

```python
import psutil
import threading
import time

class EdgePerformanceMonitor:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.monitoring = False

        # Performance metrics
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.gpu_usage = deque(maxlen=100)
        self.temperature = deque(maxlen=100)
        self.fps_history = deque(maxlen=100)

        # Performance alerts
        self.alert_thresholds = {
            'cpu': 90.0,
            'memory': 85.0,
            'gpu': 95.0,
            'temperature': 85.0,
            'fps_drop': 20.0  # % drop from target
        }

    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def monitor_loop(self):
        while self.monitoring:
            # Collect metrics
            self.collect_system_metrics()
            self.check_performance_alerts()
            time.sleep(1.0)  # 1Hz monitoring

    def collect_system_metrics(self):
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1.0)
        self.cpu_usage.append(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.percent)

        # GPU usage (NVIDIA)
        try:
            gpu_info = self.get_nvidia_gpu_info()
            self.gpu_usage.append(gpu_info['utilization'])
            self.temperature.append(gpu_info['temperature'])
        except:
            self.gpu_usage.append(0.0)
            self.temperature.append(0.0)

    def check_performance_alerts(self):
        alerts = []

        if self.cpu_usage and self.cpu_usage[-1] > self.alert_thresholds['cpu']:
            alerts.append(f"High CPU usage: {self.cpu_usage[-1]:.1f}%")

        if self.memory_usage and self.memory_usage[-1] > self.alert_thresholds['memory']:
            alerts.append(f"High memory usage: {self.memory_usage[-1]:.1f}%")

        if self.gpu_usage and self.gpu_usage[-1] > self.alert_thresholds['gpu']:
            alerts.append(f"High GPU usage: {self.gpu_usage[-1]:.1f}%")

        if self.temperature and self.temperature[-1] > self.alert_thresholds['temperature']:
            alerts.append(f"High temperature: {self.temperature[-1]:.1f}°C")

        if alerts:
            self.handle_performance_alerts(alerts)

    def handle_performance_alerts(self, alerts):
        for alert in alerts:
            print(f"⚠️  PERFORMANCE ALERT: {alert}")

        # Implement mitigation strategies
        self.apply_mitigation_strategies()

    def apply_mitigation_strategies(self):
        # Reduce model precision if GPU overloaded
        if self.gpu_usage and self.gpu_usage[-1] > 95:
            self.reduce_model_precision()

        # Throttle processing if thermal throttling
        if self.temperature and self.temperature[-1] > 85:
            self.throttle_processing()

        # Clear caches if memory high
        if self.memory_usage and self.memory_usage[-1] > 85:
            self.clear_memory_caches()

    def get_performance_report(self):
        if not self.cpu_usage:
            return "No performance data available"

        report = {
            'avg_cpu': sum(self.cpu_usage) / len(self.cpu_usage),
            'max_cpu': max(self.cpu_usage),
            'avg_memory': sum(self.memory_usage) / len(self.memory_usage),
            'max_memory': max(self.memory_usage),
            'avg_gpu': sum(self.gpu_usage) / len(self.gpu_usage),
            'max_gpu': max(self.gpu_usage),
            'avg_temp': sum(self.temperature) / len(self.temperature),
            'max_temp': max(self.temperature)
        }

        return report
```

#### **Performance Profiling**

```python
import cProfile
import pstats
from functools import wraps

class PerformanceProfiler:
    def __init__(self):
        self.profiles = {}
        self.current_profile = None

    def profile_function(self, name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if name not in self.profiles:
                    self.profiles[name] = cProfile.Profile()

                profile = self.profiles[name]
                profile.enable()

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    profile.disable()

            return wrapper
        return decorator

    def get_profile_stats(self, name):
        if name not in self.profiles:
            return None

        stats = pstats.Stats(self.profiles[name])
        stats.sort_stats('cumulative')

        # Get top 10 functions by cumulative time
        return stats.stats

    def optimize_bottlenecks(self, profile_name, threshold_ms=10):
        stats = self.get_profile_stats(profile_name)
        if not stats:
            return []

        bottlenecks = []
        for func, (cc, nc, tt, ct, callers) in stats.items():
            if ct > threshold_ms:  # Cumulative time threshold
                bottlenecks.append({
                    'function': func,
                    'cumulative_time': ct,
                    'total_calls': nc,
                    'average_time': tt / nc if nc > 0 else 0
                })

        # Sort by cumulative time
        bottlenecks.sort(key=lambda x: x['cumulative_time'], reverse=True)
        return bottlenecks

# Usage example
profiler = PerformanceProfiler()

@profiler.profile_function('inference_pipeline')
def run_inference_pipeline(frame):
    # Inference implementation
    pass

@profiler.profile_function('preprocessing')
def preprocess_frame(frame):
    # Preprocessing implementation
    pass
```

## 🐳 Containerized Deployment

### Docker for Edge

#### **Multi-Stage Builds**

```dockerfile
# Dockerfile for edge AI deployment
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.9-dev \
    python3-pip \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Build application
WORKDIR /app
COPY . .
RUN python3 setup.py build

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built application
COPY --from=builder /usr/local/lib/python3.9/dist-packages/ /usr/local/lib/python3.9/dist-packages/
COPY --from=builder /app /app

WORKDIR /app

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose ROS 2 ports
EXPOSE 5555 5556

# Run application
CMD ["python3", "main.py"]
```

#### **Docker Compose for Multi-Service**

```yaml
# docker-compose.yml for edge deployment
version: '3.8'

services:
  edge-ai:
    build:
      context: .
      dockerfile: Dockerfile.edge
    image: humanoid-robotics/edge-ai:latest
    container_name: edge-ai-node

    # GPU access
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    # Environment
    environment:
      - ROS_DOMAIN_ID=42
      - MODEL_PATH=/models/optimized
      - LOG_LEVEL=INFO

    # Volumes
    volumes:
      - ./models:/models:ro
      - ./config:/config:ro
      - /dev:/dev
      - /tmp:/tmp

    # Networks
    networks:
      - robot-network

    # Privileges for hardware access
    privileged: true

    # Restart policy
    restart: unless-stopped

  ros2-bridge:
    image: osrf/ros:humble-desktop
    container_name: ros2-bridge

    environment:
      - ROS_DOMAIN_ID=42

    volumes:
      - /dev:/dev
      - /tmp:/tmp

    networks:
      - robot-network

    privileged: true
    restart: unless-stopped

  performance-monitor:
    build:
      context: ./monitoring
      dockerfile: Dockerfile.monitor
    container_name: perf-monitor

    environment:
      - EDGE_AI_HOST=edge-ai

    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./logs:/logs

    networks:
      - robot-network

    depends_on:
      - edge-ai

    restart: unless-stopped

networks:
  robot-network:
    driver: bridge

volumes:
  models:
  config:
  logs:
```

## 🔧 Deployment Strategies

### Blue-Green Deployment

#### **Zero-Downtime Updates**

```python
class BlueGreenDeployment:
    def __init__(self, blue_config, green_config):
        self.blue_config = blue_config
        self.green_config = green_config
        self.active_side = 'blue'

    def deploy_update(self, new_image_path, health_check_port=8080):
        # Deploy to inactive side
        inactive_side = 'green' if self.active_side == 'blue' else 'blue'
        config = self.green_config if inactive_side == 'green' else self.blue_config

        print(f"Deploying to {inactive_side} side...")

        # Stage 1: Deploy new version
        self.deploy_to_side(inactive_side, new_image_path)

        # Stage 2: Health check
        if self.health_check(inactive_side, health_check_port):
            print(f"Health check passed for {inactive_side}")

            # Stage 3: Switch traffic
            self.switch_traffic(inactive_side)

            # Stage 4: Verify active side
            if self.verify_active():
                print(f"Successfully switched to {inactive_side}")
                self.active_side = inactive_side
                return True
            else:
                print(f"Verification failed, rolling back...")
                self.rollback()
                return False
        else:
            print(f"Health check failed for {inactive_side}")
            return False

    def health_check(self, side, port, timeout=30):
        # Perform comprehensive health check
        checks = [
            self.check_container_running(side),
            self.check_api_health(side, port),
            self.check_gpu_access(side),
            self.check_ros_connectivity(side)
        ]

        return all(checks)

    def check_container_running(self, side):
        try:
            container_name = f"robotics-{side}"
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() == "true"
        except:
            return False

    def check_api_health(self, side, port):
        try:
            # Make HTTP request to health endpoint
            import requests
            response = requests.get(
                f"http://localhost:{port}/health",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def check_gpu_access(self, side):
        try:
            # Execute command in container to check GPU access
            container_name = f"robotics-{side}"
            result = subprocess.run([
                "docker", "exec", container_name,
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
            ], capture_output=True, text=True, timeout=10)

            return result.returncode == 0 and "NVIDIA" in result.stdout
        except:
            return False

    def switch_traffic(self, new_active_side):
        # Update load balancer configuration
        self.update_load_balancer(new_active_side)

        # Update DNS if needed
        self.update_dns(new_active_side)

        print(f"Traffic switched to {new_active_side}")
```

### Canary Deployments

#### **Gradual Rollout**

```python
class CanaryDeployment:
    def __init__(self, total_instances=10):
        self.total_instances = total_instances
        self.canary_instances = 0
        self.stable_instances = total_instances

    def gradual_canary(self, new_image, stages=[0.1, 0.25, 0.5, 1.0]):
        for stage in stages:
            canary_count = int(self.total_instances * stage)
            stable_count = self.total_instances - canary_count

            print(f"Deploying canary stage: {stage:.0%} ({canary_count} instances)")

            # Deploy canary instances
            if not self.deploy_canary(new_image, canary_count):
                print(f"Canary deployment failed at stage {stage:.0%}")
                self.rollback_canary()
                return False

            # Monitor and validate
            if not self.monitor_canary(duration=300):  # 5 minutes
                print(f"Canary monitoring failed at stage {stage:.0%}")
                self.rollback_canary()
                return False

            # Promote canary to stable
            self.promote_canary(canary_count)

        print("Canary deployment completed successfully!")
        return True

    def monitor_canary(self, duration=300):
        # Monitor canary instances for success criteria
        start_time = time.time()

        while time.time() - start_time < duration:
            metrics = self.collect_metrics()

            # Check success criteria
            if not self.check_success_criteria(metrics):
                return False

            time.sleep(30)  # Check every 30 seconds

        return True

    def check_success_criteria(self, metrics):
        criteria = {
            'error_rate': 0.01,  # 1%
            'response_time_p95': 100,  # 100ms
            'cpu_usage': 80,  # 80%
            'memory_usage': 85  # 85%
        }

        for metric, threshold in criteria.items():
            if metrics.get(metric, 0) > threshold:
                print(f"Canary failed: {metric} = {metrics[metric]} > {threshold}")
                return False

        return True

    def collect_metrics(self):
        # Collect metrics from monitoring system
        # This would integrate with Prometheus, Grafana, etc.
        return {
            'error_rate': self.get_error_rate(),
            'response_time_p95': self.get_response_time_p95(),
            'cpu_usage': self.get_cpu_usage(),
            'memory_usage': self.get_memory_usage()
        }
```

## 📋 Practical Implementation

### Complete Edge AI System

#### **Main Application**

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class EdgeAIRobot(Node):
    def __init__(self):
        super().__init__('edge_ai_robot')

        # Initialize components
        self.inference_engine = self.initialize_inference()
        self.performance_monitor = self.initialize_monitoring()
        self.resource_manager = self.initialize_resources()

        # ROS 2 interfaces
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.process_image, 10)
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

        self.get_logger().info('Edge AI Robot initialized')

    def initialize_inference(self):
        # Load optimized model
        model_path = "/models/optimized/humanoid_perception.trt"
        engine = TensorRTEngine(model_path)
        return engine

    def initialize_monitoring(self):
        monitor = EdgePerformanceMonitor(target_fps=30)
        monitor.start_monitoring()
        return monitor

    def initialize_resources(self):
        manager = ResourceManager()
        manager.setup_gpu_profile()
        return manager

    def process_image(self, msg):
        start_time = time.time()

        try:
            # Convert ROS image to tensor
            image_tensor = self.ros_to_tensor(msg)

            # Run inference
            detections = self.inference_engine.infer(image_tensor)

            # Process detections for robot control
            cmd = self.detections_to_control(detections)

            # Publish control command
            self.cmd_vel_pub.publish(cmd)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.update_performance_metrics(processing_time)

        except Exception as e:
            self.get_logger().error(f"Processing failed: {e}")

    def update_performance_metrics(self, processing_time):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        if elapsed_time > 10.0:  # Report every 10 seconds
            fps = self.frame_count / elapsed_time
            avg_processing_time = processing_time / self.frame_count

            self.get_logger().info(
                f"Performance: {fps:.1f} FPS, "
                f"Avg processing: {avg_processing_time*1000:.1f}ms"
            )

            # Get system metrics
            report = self.performance_monitor.get_performance_report()
            self.get_logger().info(
                f"System: CPU {report['avg_cpu']:.1f}%, "
                f"Memory {report['avg_memory']:.1f}%, "
                f"GPU {report['avg_gpu']:.1f}%"
            )

            # Reset counters
            self.frame_count = 0
            self.start_time = time.time()

def main():
    rclpy.init()

    try:
        node = EdgeAIRobot()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 🎯 Chapter Summary

### Key Takeaways

1. **Edge AI Architecture**
   - Design systems for resource-constrained environments
   - Balance computational load between edge and cloud
   - Implement real-time inference pipelines
   - Ensure robust error handling and fallback mechanisms

2. **Model Optimization**
   - Apply quantization and pruning techniques
   - Use TensorRT for optimal GPU performance
   - Implement batch processing for efficiency
   - Monitor and maintain model accuracy

3. **Deployment Strategies**
   - Use containerization for consistent environments
   - Implement blue-green and canary deployments
   - Design for zero-downtime updates
   - Automate deployment and monitoring

4. **Performance Management**
   - Monitor system resources in real-time
   - Implement adaptive load balancing
   - Profile and optimize bottlenecks
   - Plan for thermal and power constraints

### Next Steps

With edge deployment skills mastered, you're ready for Quarter 4: **Multimodal AI and Human-Robot Interaction**. In the final quarter, you'll integrate all previous skills to create truly intelligent humanoid robots that can understand and interact with humans through multiple modalities including vision, language, and voice.

---

**Ready to proceed?** Continue with [Quarter 4: Multimodal AI and Human-Robot Interaction](../quarter-4/index.md) to create the ultimate humanoid robotics experience! 🤖✨

**Pro Tip**: Edge deployment is where simulation meets reality. Test thoroughly, monitor continuously, and always have fallback mechanisms. The skills you've developed in this chapter are essential for deploying AI-powered robots in real-world applications! 🚀