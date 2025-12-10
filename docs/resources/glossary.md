---
title: "Glossary"
sidebar_label: "Glossary"
sidebar_position: 2
---

# Glossary

## Comprehensive Terminology Guide

This glossary provides definitions for key terms, acronyms, and concepts used throughout the humanoid robotics educational guide.

## 🤖 Core Robotics Concepts

### A

**Actuator**: A device that converts energy into mechanical motion. In humanoid robots, actuators control joint movement, allowing for realistic human-like motion.

**Articulation**: The arrangement of joints and movable segments that make up a robot's body structure.

**Autonomous Robot**: A robot capable of performing tasks without direct human control, using sensors and programming to make decisions.

**Autonomy**: The ability of a robot to operate independently without human intervention, making its own decisions based on sensor input and programming.

### B

**Base Link**: The stationary or root component of a robot's kinematic chain to which all other links are connected.

**Base Frame**: A coordinate frame attached to the base of a robot, serving as a reference for all other coordinate transformations.

**Bidirectional Communication**: Two-way data exchange between systems, enabling both sending and receiving information.

### C

**Cartesian Coordinates**: A system that specifies each point uniquely in a plane by a set of numerical coordinates.

**Compliance**: The ability of a robot to yield to external forces, important for safe human-robot interaction.

**Configuration Space**: The set of all possible positions and orientations a robot can achieve.

**Coordinate Frame**: A reference system used to describe the position and orientation of objects in 3D space.

**Cyber-Physical System**: Systems that integrate computing, networking, and physical processes, such as robots.

### D

**Degree of Freedom (DoF)**: The number of independent parameters that define the configuration of a system. Humanoid robots typically have 30+ degrees of freedom.

**Dynamics**: The study of forces and torques that cause motion, essential for robot motion planning and control.

**Dead Reckoning**: The process of estimating current position based on previously determined positions and velocity measurements.

### E

**End Effector**: The device at the end of a robot arm that interacts with the environment (e.g., gripper, tool, or sensor).

**Encoder**: A sensor that measures position, velocity, or other motion parameters for feedback control.

**Environment Modeling**: The process of creating a representation of the robot's surroundings for navigation and planning.

**Extrinsic Parameters**: Camera calibration parameters that describe the camera's position and orientation in world coordinates.

### F

**Forward Kinematics**: The problem of determining the position and orientation of the robot's end effector given joint angles or displacements.

**Force/Torque Sensor**: A sensor that measures contact forces and torques, crucial for safe human-robot interaction.

### G

**Gait**: The pattern of locomotion for walking or running robots, mimicking human walking patterns.

**Geometric Primitives**: Basic geometric shapes (points, lines, planes) used in robot motion planning and collision detection.

**Graph SLAM**: Simultaneous Localization and Mapping using graph-based representations.

### H

**Humanoid Robot**: A robot with a body shape resembling that of a human, typically with a torso, two arms, two legs, and a head.

**Hybrid Automaton**: A mathematical model that combines discrete and continuous dynamics, useful for robot behavior specification.

### I

**IMU (Inertial Measurement Unit)**: An electronic device that measures and reports a body's specific force, angular rate, and magnetic field.

**Intrinsic Parameters**: Camera calibration parameters that describe internal camera characteristics (focal length, principal point, distortion).

**Inverse Kinematics**: The problem of determining joint angles or displacements that achieve a desired end effector position and orientation.

### J

**Joint**: A connection between two links in a robot that allows relative motion, typically rotational or translational.

**Joint Space**: The space defined by all possible joint configurations of a robot.

**Jacobians**: Matrices that relate joint velocities to end effector velocities, used in robot control and motion planning.

### K

**Kalman Filter**: An algorithm that uses a series of measurements observed over time to produce estimates of unknown variables.

**Kinematic Chain**: A connected series of rigid links and joints forming the structure of a robot manipulator.

**Kinematics**: The study of motion without considering forces, describing how objects move.

### L

**LiDAR (Light Detection and Ranging)**: A sensor that uses laser pulses to measure distances to create 3D point clouds of environments.

**Linear Algebra**: The branch of mathematics concerning vector spaces and linear mappings between them, fundamental to robotics computations.

**Localization**: The process of determining a robot's position and orientation within its environment.

### M

**Manipulator**: A robot arm mechanism consisting of a series of links connected by joints.

**Map Building**: The process of creating a representation of an environment for robot navigation.

**Mobile Manipulation**: The combination of mobile robot platforms with manipulator arms.

**Mobile Robot**: A robot capable of locomotion in an environment, often using wheels, legs, or tracks.

### O

**Obstacle Avoidance**: The ability of a robot to detect and avoid collisions with obstacles in its path.

**Odometry**: The use of motion sensors to estimate change in position over time.

**Operating Space**: The volume of space reachable by the robot's end effector.

**Orientation**: The angular position of an object relative to a reference frame, often described using roll, pitch, and yaw angles.

### P

**Path Planning**: The process of finding a collision-free path from a start configuration to a goal configuration.

**PID Controller**: A proportional-integral-derivative controller, widely used in robot control systems.

**Point Cloud**: A set of 3D points representing the external surface of an object or environment, often from LiDAR sensors.

**Pose**: The combination of position and orientation of an object in 3D space.

**Position Control**: Robot control that focuses on achieving specific positions without regard to path or trajectory.

### Q

**Quaternion**: A four-dimensional number system used to represent rotations in 3D space, avoiding gimbal lock issues.

### R

**Redundant Robot**: A robot with more degrees of freedom than required for its task, providing flexibility in motion planning.

**Reference Frame**: A coordinate system used as a reference for describing positions and orientations.

**Robot Operating System (ROS)**: A flexible framework for writing robot software, providing hardware abstraction, device drivers, libraries, visualizers, and more.

**Robot State**: The configuration of a robot, including joint positions, velocities, and sensor readings.

### S

**Sensor Fusion**: The process of combining data from multiple sensors to improve perception and understanding.

**SLAM (Simultaneous Localization and Mapping)**: The computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it.

**Singularity**: A robot configuration where one or more degrees of freedom are lost, typically occurring at the workspace boundaries.

**Subsumption Architecture**: A behavioral architecture for robot control that uses layers of task-achieving behaviors.

### T

**Task Space**: The space in which the robot's end effector operates, often Cartesian coordinates.

**Teleoperation**: Remote control of a robot by a human operator, often through visual feedback.

**Trajectory**: A time-parameterized path that includes timing information for motion between points.

**Transformation**: Mathematical operation that converts coordinates from one reference frame to another.

### V

**Velocity Control**: Robot control that focuses on controlling speed and direction of motion rather than position.

**Visual Servoing**: A control technique that uses computer vision feedback to control robot motion relative to visual targets.

### W

**Workspace**: The total volume of space that a robot's end effector can reach.

**World Frame**: A fixed, global coordinate frame used as the ultimate reference for all robot operations.

## 🧠 Computer Vision Terms

### A

**Active Vision**: Computer vision systems that actively control camera parameters like focus, zoom, or viewpoint.

**Affine Transformation**: A linear transformation that preserves parallelism but not necessarily angles and distances.

**Augmented Reality**: Technology that superimposes computer-generated images on a user's view of the real world.

### C

**Camera Calibration**: The process of estimating the parameters of a camera model to correct for lens distortion and other effects.

**Convolutional Neural Network (CNN)**: A deep learning architecture particularly effective for image processing tasks.

**Corner Detection**: Computer vision algorithms for detecting points in images with high curvature, useful for feature tracking.

### D

**Depth Map**: A 2D image where each pixel's value represents the depth (distance) from the camera to the corresponding point.

**Digital Image Processing**: The use of computer algorithms to perform image processing on digital images.

### E

**Epipolar Geometry**: The geometry of stereo vision, describing the geometric relationships between two camera views.

### F

**Feature Detection**: Computer vision algorithms for identifying distinctive points, edges, or regions in images.

**Feature Matching**: The process of finding corresponding features between different images.

**Feature Descriptor**: A vector of values that describes an image feature, enabling comparison between features.

**Focal Length**: The distance from the camera's lens to its sensor, a key parameter in camera calibration.

### G

**Gaussian Blur**: An image processing technique that uses a Gaussian function to blur an image, reducing noise and detail.

**Geometric Transformation**: Operations that change the geometric properties of images, such as translation, rotation, scaling, and shearing.

### H

**Harris Corner Detector**: An algorithm for corner detection in computer vision, widely used for feature tracking.

**Histogram Equalization**: An image processing technique that improves contrast by adjusting the image's histogram.

### I

**Image Segmentation**: The process of partitioning a digital image into multiple segments to simplify or change the representation of the image.

**Intrinsic Parameters**: Internal camera parameters such as focal length, principal point, and lens distortion coefficients.

### K

**Keypoint Detection**: Computer vision algorithms for detecting distinctive points in images that can be reliably tracked across multiple views.

### L

**Lens Distortion**: Optical aberrations that cause straight lines to appear curved in images.

### M

**Machine Vision**: The technology and methods used to provide imaging-based automatic inspection, process control, and robot guidance.

### O

**Object Detection**: Computer vision algorithms that locate instances of semantic objects of a certain class in digital images.

### P

**Panoramic Stitching**: The process of combining multiple photographic images with overlapping fields of view to produce a single panoramic image.

### R

**Rectification**: The process of correcting image distortion to produce undistorted images.

### S

**Scale-Invariant Feature Transform (SIFT)**: An algorithm in computer vision to detect and describe local features in images.

### V

**Visual Odometry**: The process of estimating the position and orientation of a vehicle by analyzing the sequence of images captured by its cameras.

## 🔧 Artificial Intelligence Terms

### A

**Activation Function**: A mathematical function that determines the output of a neural network node based on its input.

**Artificial General Intelligence (AGI)**: The hypothetical ability of an AI system to understand or learn any intellectual task that a human being can.

**Artificial Neural Network**: A computing system inspired by biological neural networks that constitute animal brains.

**Attention Mechanism**: A component in neural networks that allows the model to focus on different parts of the input when generating outputs.

### B

**Backpropagation**: A method used in artificial neural networks to calculate the gradient of the loss function with respect to the weights of the network.

**Batch Normalization**: A technique for training deep neural networks that normalizes the activations between layers.

**Bias**: A parameter in neural networks that allows shifting the activation function up or down.

### C

**Convolutional Neural Network (CNN)**: A deep learning architecture particularly effective for image processing tasks using convolutional layers.

**Cross-Entropy Loss**: A loss function used in classification tasks to measure the difference between predicted and actual probabilities.

**Cross-Validation**: A resampling procedure used to evaluate machine learning models on limited data samples.

### D

**Deep Learning**: A subset of machine learning that uses artificial neural networks with multiple layers.

**Dropout**: A regularization technique for reducing overfitting in neural networks by randomly dropping units during training.

### E

**Epoch**: One complete pass through the entire training dataset during neural network training.

**Embedding**: A learned representation of categorical variables in a continuous vector space.

### F

**Feature Engineering**: The process of using domain knowledge to select and transform variables when creating machine learning models.

**Feedforward Neural Network**: An artificial neural network where connections between the nodes do not form a cycle.

**Fine-Tuning**: The process of taking a pretrained model and further training it on a specific task.

### G

**Gradient Descent**: An optimization algorithm used to minimize the loss function of a neural network by iteratively adjusting the weights.

### H

**Hidden Layer**: A layer of neurons in a neural network that are not directly connected to input or output.

**Hyperparameter**: A parameter whose value is set before the learning process begins, as opposed to model parameters learned during training.

### L

**Learning Rate**: A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.

**Loss Function**: A function that maps the difference between predicted and actual values to a real number that represents the cost of making a prediction.

### M

**Machine Learning**: A field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.

**Multi-Task Learning**: A learning paradigm where multiple related tasks are learned simultaneously, sharing knowledge across tasks.

### N

**Neural Network**: A computing system inspired by biological neural networks.

**Normalization**: The process of rescaling input data to have a standard distribution, improving model training.

### O

**Overfitting**: A modeling error where the model learns the training data too well and performs poorly on new, unseen data.

### R

**Recurrent Neural Network (RNN)**: A type of neural network where connections between nodes form a directed graph along a temporal sequence.

**Regularization**: Techniques used to prevent overfitting by adding constraints or penalties to the loss function.

**Reinforcement Learning**: An area of machine learning concerned with how software agents ought to take actions in an environment to maximize reward.

### S

**Supervised Learning**: A type of machine learning where the algorithm learns from labeled training data.

### T

**Transfer Learning**: A machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

**Transformer**: A neural network architecture that uses self-attention mechanisms, revolutionizing natural language processing.

### U

**Underfitting**: A modeling error where a model is too simple to capture the underlying structure of the data.

**Unsupervised Learning**: A type of machine learning that looks for previously undetected patterns in a dataset with no pre-existing labels.

### V

**Validation Set**: A portion of the dataset used to assess the performance of a trained model during the training process.

## 🎮 Simulation Terms

### A

**Agent**: An entity that can perceive its environment and act upon it, often used in simulation environments.

**Asset**: Any resource used in a simulation, such as models, textures, or environments.

### C

**Collision Detection**: The computational problem of detecting the intersection of two or more objects.

**Digital Twin**: A virtual representation of a physical object or system that serves as its digital counterpart.

### E

**Environment**: The virtual world in which a simulation takes place, including all objects, lighting, and physics parameters.

### F

**Finite Element Analysis**: Numerical method for solving partial differential equations, often used in simulation.

### G

**Game Engine**: Software framework designed for creating video games, often adapted for robotics simulation.

### H

**Haptic Feedback**: Technology that simulates the sense of touch through force, vibration, or motion.

### I

**Inverse Dynamics**: The process of calculating the torques and forces that cause a particular motion.

### J

**Joint Simulation**: Multi-body dynamics simulation that involves analyzing the motion of interconnected bodies.

### K

**Kinematic Simulation**: Motion simulation without considering forces or torques.

### L

**Level of Detail (LOD)**: A technique for reducing the complexity of 3D models when they are far from the viewer.

### M

**Material Properties**: Physical properties of objects used in simulation, such as density, elasticity, and friction.

**Mesh Generation**: The process of creating polygonal meshes from 3D models for simulation.

### N

**Numerical Integration**: Approximate calculation of definite integrals, used in simulation physics engines.

### O

**Occupancy Grid**: A grid-based representation of environment space that marks areas as occupied, free, or unknown.

### P

**Physics Engine**: Software that simulates physical phenomena such as gravity, collision, and material properties.

### R

**Rendering**: The process of generating an image from a 2D or 3D model using computer graphics.

### S

**Simulation Loop**: The main loop in simulation software that advances time and updates physics calculations.

**Sim2Real**: The process of transferring simulation results to real-world applications.

### T

**Time Step**: The discrete interval of time in a simulation during which physics calculations are updated.

**Trajectory Optimization**: The process of finding optimal paths or motions for robots.

### V

**Virtual Reality (VR)**: A computer-generated simulation of a three-dimensional environment that can be interacted with.

### W

**World Model**: A representation of the environment used by robots for planning and decision-making.

## 🎛️ Hardware Terms

### A

**Actuator**: A device that converts energy into mechanical motion to control a mechanism or system.

**Analog-to-Digital Converter (ADC)**: A device that converts analog signals to digital signals.

### B

**Battery Management System (BMS)**: Electronics that manage the charging and discharging of rechargeable batteries.

**Brushless DC Motor**: An electric motor powered by direct current (DC) that uses electronic commutation instead of mechanical brushes.

### C

**Controller**: Electronic device that manages, commands, directs, or regulates the behavior of other devices or systems.

**CPU (Central Processing Unit)**: The electronic circuitry within a computer that carries out the instructions of a computer program.

### D

**Digital-to-Analog Converter (DAC)**: A device that converts digital signals into analog signals.

**Driver**: Software that allows a computer to interact with and control hardware devices.

**Dynamic Range**: The ratio between the largest and smallest values that a sensor can measure.

### E

**Encoder**: A device that converts motion or position into electrical signals, often used for joint position sensing.

**Ethernet**: A family of computer networking technologies commonly used in local area networks (LANs).

### F

**Field-Programmable Gate Array (FPGA)**: An integrated circuit designed to be configured by a customer after manufacturing.

**Force Sensor**: A device that measures force and torque, providing feedback for robot control.

### G

**GPIO (General Purpose Input/Output)**: Generic pins on an integrated circuit whose behavior can be controlled by the user at runtime.

**Gyroscope**: A sensor that measures angular velocity, used for orientation sensing in robots.

### I

**Inertial Measurement Unit (IMU)**: An electronic device that measures and reports a body's specific force, angular rate, and magnetic field.

### L

**LiDAR**: Light Detection and Ranging sensor that uses laser pulses to measure distances.

### M

**Microcontroller**: A compact integrated circuit designed for embedded applications.

**Motor Driver**: Electronic circuit or set of circuits that controls the speed and torque of an electric motor.

### P

**Potentiometer**: A three-terminal resistor with a sliding contact that forms an adjustable voltage divider.

**Power Supply**: An electrical device that supplies electric power to an electrical load.

### R

**Raspberry Pi**: A series of small single-board computers developed in the UK.

**Real-Time Operating System (RTOS)**: An operating system intended to serve real-time applications that process data as it comes in, typically without buffer delays.

**Resolution**: The smallest increment a sensor can measure, often expressed in bits or units.

### S

**Servo Motor**: A rotary actuator or linear actuator that allows for precise control of angular or linear position.

### T

**Torque**: The rotational equivalent of linear force, important for robot actuator specification.

### U

**USB (Universal Serial Bus)**: An industry standard that establishes specifications for cables and connectors for connection, communication, and power supply between computers.

## 📡 Communication Terms

### A

**API (Application Programming Interface)**: A set of definitions and protocols for building and integrating application software.

### C

**CAN Bus**: Controller Area Network, a robust vehicle bus standard designed to allow microcontrollers and devices to communicate with each other.

**Client-Server Architecture**: A distributed application structure that separates tasks between service providers (servers) and service requesters (clients).

### D

**Data Rate**: The number of bits transmitted per unit of time in a communication system.

### E

**Ethernet**: A family of computer networking technologies commonly used in local area networks (LANs).

### F

**Firewall**: A network security system that monitors and controls incoming and outgoing network traffic.

### H

**Handshaking**: An automated process of negotiation that sets parameters of a communication channel between two entities before normal communication begins.

### I

**IP Address**: A numerical label assigned to each device connected to a computer network that uses the Internet Protocol for communication.

### M

**Message Queueing**: A method of communication between applications or services where messages are stored until the receiving application is ready to process them.

### N

**Network Protocol**: A set of rules and conventions for communication between network devices.

### P

**Packet**: A formatted unit of data carried by a packet-switched network.

**Protocol**: A system of rules that allow two or more entities of a communications system to transmit information.

### R

**Real-Time Communication**: Communication with timing constraints, where late or early messages are considered incorrect.

**Router**: A networking device that forwards data packets between computer networks.

### S

**Socket**: An endpoint for sending or receiving data across a computer network.

### T

**TCP (Transmission Control Protocol)**: A connection-oriented protocol that provides reliable, ordered, and error-checked delivery of a stream of bytes.

### U

**UDP (User Datagram Protocol)**: A connectionless communication protocol that provides minimal message transaction processing.

### W

**Wireless Communication**: The transfer of information over a distance without using electrical conductors or wires.

## 🔊 Signal Processing Terms

### A

**Amplitude**: The maximum displacement or distance moved by a point on a vibrating body or wave measured from its equilibrium position.

### B

**Bandwidth**: The range of frequencies between lower and upper cutoff frequencies, usually defined by the points at which the signal drops to specified percentages.

**Bit Depth**: The number of bits of information in each sample of digital audio.

### C

**Codec**: Device or program that compresses and decompresses digital audio data.

### D

**Digital Signal Processing (DSP)**: The use of digital processing to perform a wide variety of signal processing operations.

### F

**Fourier Transform**: A mathematical transform that decomposes functions depending on space or time into functions depending on spatial or temporal frequency.

### F

**Frequency**: The number of occurrences of a repeating event per unit of time.

### H

**Harmonic**: A wave with a frequency that is an integer multiple of the fundamental frequency.

### N

**Noise**: Unwanted random additions to a signal that obscure the information content.

### Q

**Quantization**: The process of mapping a large set of input values to a smaller set, often used in analog-to-digital conversion.

### R

**Resampling**: The process of changing the sampling rate of a signal.

### S

**Sampling**: The reduction of a continuous-time signal to a discrete-time signal.

### T

**Time Domain**: The analysis of mathematical functions, physical signals, or time series with respect to time.

### V

**Voltage**: An electric potential difference between two points, often used to represent audio signal amplitude.

---

**📖 Learning robotics is an ongoing journey!** This glossary will continue to expand as new technologies and concepts emerge in the field of humanoid robotics and artificial intelligence.

**💡 Pro Tip**: Understanding these fundamental terms is crucial for effective communication in robotics teams and for reading technical documentation and research papers.