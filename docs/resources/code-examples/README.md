# Code Examples Repository

This directory contains code examples and implementation guides referenced throughout the humanoid robotics educational guide.

## 📂 Directory Structure

```
code-examples/
├── ros2/
│   ├── basic_nodes/
│   ├── publishers/
│   └── services/
├── computer_vision/
│   ├── opencv_basics/
│   ├── object_detection/
│   └── image_processing/
├── deep_learning/
│   ├── pytorch_models/
│   ├── tensorflow_models/
│   └── training_scripts/
├── robotics/
│   ├── motion_planning/
│   ├── control/
│   └── simulation/
└── integration/
    ├── multimodal/
    ├── voice_control/
    └── hardware_integration/
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- ROS 2 Humble
- OpenCV 4.x
- PyTorch 1.9+ (optional)
- TensorFlow 2.x (optional)

### Installation

```bash
# Clone this repository
git clone https://github.com/humanoid-robotics/code-examples
cd code-examples

# Install Python dependencies
pip install -r requirements.txt

# Set up ROS 2 environment
source /opt/ros/humble/setup.bash
```

## 📚 Example Categories

### ROS 2 Examples

#### Basic Publishers
- **`ros2/basic_nodes/publisher.py`**: Simple message publisher
- **`ros2/basic_nodes/subscriber.py`**: Message subscriber implementation
- **`ros2/basic_nodes/service_server.py`**: ROS 2 service implementation

#### Services and Actions
- **`ros2/services/add_two_ints_server.py`**: Service example for adding integers
- **`ros2/actions/fibonacci_server.py`**: Action server implementation

### Computer Vision Examples

#### OpenCV Fundamentals
- **`computer_vision/opencv_basics/image_processing.py`**: Basic image operations
- **`computer_vision/opencv_basics/video_processing.py`**: Video stream processing
- **computer_vision/opencv_basics/camera_calibration.py`: Camera calibration procedure

#### Object Detection
- **`computer_vision/object_detection/yolo_detector.py`**: YOLO object detection
- **computer_vision/object_detection/hog_detector.py**: Histogram of Oriented Gradients detector

### Deep Learning Examples

#### PyTorch Models
- **`deep_learning/pytorch_models/simple_cnn.py`**: Convolutional neural network
- **`deep_learning/pytorch_models/lstm_network.py`: Recurrent neural network
- **`deep_learning/pytorch_models/transformer_model.py`: Transformer architecture

#### Training Scripts
- **`deep_learning/training_scripts/train_classifier.py`**: Model training pipeline
- **deep_learning/training_scripts/data_loader.py`: Custom data loader

### Robotics Examples

#### Motion Planning
- **`robotics/motion_planning/astar_planner.py`**: A* path planning algorithm
- **robotics/motion_planning/rrt_planner.py**: RRT planning algorithm
- **robotics/motion_planning/dwa_planner.py**: Dynamic Window Approach

#### Control Systems
- `robotics/control/pid_controller.py`: PID implementation
- `robotics/control/trajectory_tracking.py`: Trajectory following
- `robotics/control/kinematics.py`: Forward/inverse kinematics

### Integration Examples

#### Multimodal AI
- `integration/multimodal/vision_language_fusion.py`: Vision-language model fusion
- `integration/multimodal/audio_visual_fusion.py`: Audio-visual sensor fusion

#### Voice Control
- `integration/voice_control/speech_to_command.py`: Speech recognition
- `integration/voice_control/text_to_speech.py`: Text-to-speech synthesis

#### Hardware Integration
- `integration/hardware_integration/sensor_reading.py`: Sensor data acquisition
- `integration/hardware_integration/actuator_control.py`: Actuator control

## 🔧 Usage Instructions

### Running Examples

Each example includes:

1. **Source code** with comprehensive comments
2. **Requirements file** listing dependencies
3. **README.md** with setup and usage instructions
4. **Example data** where applicable

To run an example:

```bash
cd path/to/example
python example.py
```

### ROS 2 Examples

```bash
# Build packages
colcon build --packages-select my_robot_package

# Source environment
source install/setup.bash

# Run node
ros2 run my_robot_package example_node
```

### Deep Learning Examples

```bash
# Install PyTorch dependencies
pip install torch torchvision

# Run training script
python train_model.py --data_path /path/to/data
```

## 🧪 Testing

Each example includes unit tests:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_example.py
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add your example with proper documentation
4. Include tests where appropriate
5. Submit a pull request

## 📄 License

All code examples are released under the MIT License. See the LICENSE file for details.

## 🆘 Support

For issues or questions:

- Open an issue on GitHub
- Check the documentation
- Review the main educational guide for context

## 📞 Contact

- Educational Guide: [Link to main repo]
- Issues: [Link to issues page]
- Discussions: [Link to discussions]