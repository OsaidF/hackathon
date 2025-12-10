---
title: "Chapter 5: Hardware Introduction"
sidebar_label: "5. Hardware Introduction"
sidebar_position: 5
---

# Chapter 5: Hardware Introduction

## Bridging Software and Physical Reality

Hardware is the physical foundation upon which all robotics software operates. Understanding hardware components, their capabilities, limitations, and interfaces is essential for designing effective robotic systems. This chapter introduces the fundamental hardware components that make up modern robots and provides practical guidance for working with them.

## 🖥️ Computing Platforms

### Single Board Computers (SBCs)

#### Raspberry Pi
- **Use Case**: Education, prototyping, lightweight robotics
- **Pros**: Low cost, large community, GPIO pins
- **Cons**: Limited processing power, non-real-time
- **Typical Applications**: Small mobile robots, sensor nodes

```python
# Raspberry Pi GPIO control example
import RPi.GPIO as GPIO
import time

class MotorController:
    def __init__(self):
        # Setup GPIO pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT)  # Motor A enable
        GPIO.setup(27, GPIO.OUT)  # Motor B enable
        GPIO.setup(22, GPIO.OUT)  # Direction pin 1
        GPIO.setup(23, GPIO.OUT)  # Direction pin 2

        # PWM for speed control
        self.pwm_a = GPIO.PWM(17, 1000)  # 1kHz frequency
        self.pwm_b = GPIO.PWM(27, 1000)

        self.pwm_a.start(0)
        self.pwm_b.start(0)

    def set_motor_speed(self, motor_a_speed, motor_b_speed):
        """Set motor speeds (-100 to 100)"""
        # Motor A
        if motor_a_speed > 0:
            GPIO.output(22, GPIO.HIGH)
            GPIO.output(23, GPIO.LOW)
        else:
            GPIO.output(22, GPIO.LOW)
            GPIO.output(23, GPIO.HIGH)

        self.pwm_a.ChangeDutyCycle(abs(motor_a_speed))

        # Motor B
        if motor_b_speed > 0:
            GPIO.output(22, GPIO.HIGH)
            GPIO.output(23, GPIO.LOW)
        else:
            GPIO.output(22, GPIO.LOW)
            GPIO.output(23, GPIO.HIGH)

        self.pwm_b.ChangeDutyCycle(abs(motor_b_speed))

    def cleanup(self):
        """Clean up GPIO resources"""
        self.pwm_a.stop()
        self.pwm_b.stop()
        GPIO.cleanup()
```

#### NVIDIA Jetson
- **Use Case**: AI-powered robots, computer vision applications
- **Pros**: GPU acceleration, CUDA support, high performance
- **Cons**: Higher cost, more power consumption
- **Typical Applications**: Autonomous navigation, object detection, AI robots

```python
# Jetson GPU acceleration example
import jetson.inference
import jetson.utils

class VisionProcessor:
    def __init__(self):
        # Load neural network for object detection
        self.net = jetson.inference.detectNet(
            "ssd-mobilenet-v2",
            threshold=0.5
        )

        # Camera setup
        self.camera = jetson.utils.gstCamera(
            1280, 720, "/dev/video0"
        )

        self.display = jetson.utils.glDisplay()

    def process_frame(self):
        """Process camera frame for object detection"""
        # Capture image
        img, width, height = self.camera.CaptureRGBA()

        # Detect objects
        detections = self.net.Detect(img, width, height)

        # Process detections
        objects = []
        for detection in detections:
            obj_info = {
                'class': self.net.GetClassDesc(detection.ClassID),
                'confidence': detection.Confidence,
                'bbox': (
                    detection.Left, detection.Top,
                    detection.Right, detection.Bottom
                )
            }
            objects.append(obj_info)

        # Display results
        self.display.RenderOnce(img, width, height)
        self.display.SetTitle("Object Detection | Network {:.0f} FPS".format(
            self.net.GetNetworkFPS()))

        return objects
```

#### Intel NUC / Industrial PCs
- **Use Case**: High-performance robotics, industrial applications
- **Pros**: x86 architecture, expandability, multiple OS support
- **Cons**: Higher power consumption, larger form factor
- **Typical Applications**: Robot control centers, data processing hubs

### Microcontrollers

#### Arduino / ESP32
- **Use Case**: Real-time control, sensor interfacing, actuator control
- **Pros**: Real-time performance, low power, I/O flexibility
- **Cons**: Limited processing, simple programming model
- **Typical Applications**: Motor control, sensor reading, safety systems

```cpp
// Arduino code for real-time sensor reading
class SensorController {
private:
    const int TRIG_PIN = 9;
    const int ECHO_PIN = 10;
    const int MOTOR_PIN = 6;

public:
    SensorController() {
        pinMode(TRIG_PIN, OUTPUT);
        pinMode(ECHO_PIN, INPUT);
        pinMode(MOTOR_PIN, OUTPUT);
    }

    float readDistance() {
        // Send ultrasonic pulse
        digitalWrite(TRIG_PIN, LOW);
        delayMicroseconds(2);
        digitalWrite(TRIG_PIN, HIGH);
        delayMicroseconds(10);
        digitalWrite(TRIG_PIN, LOW);

        // Measure echo time
        long duration = pulseIn(ECHO_PIN, HIGH);

        // Convert to distance (cm)
        return duration * 0.034 / 2;
    }

    void controlMotor(int speed) {
        // Speed: 0-255
        analogWrite(MOTOR_PIN, constrain(speed, 0, 255));
    }

    void update() {
        // Real-time control loop
        float distance = readDistance();

        if (distance < 20.0) {
            controlMotor(0);  // Stop if obstacle detected
        } else if (distance < 50.0) {
            controlMotor(128);  // Slow speed
        } else {
            controlMotor(255);  // Full speed
        }
    }
};

SensorController controller;

void setup() {
    Serial.begin(9600);
}

void loop() {
    controller.update();
    delay(10);  // 100 Hz control frequency
}
```

## 📡 Communication Interfaces

### USB
- **Bandwidth**: Up to 20 Gbps (USB 4.0)
- **Use Case**: High-bandwidth sensors, debugging, configuration
- **Pros**: Universal, hot-pluggable, power delivery
- **Cons**: Limited cable length, non-real-time guarantees

```python
# USB device communication example
import pyusb
import time

class USBDevice:
    def __init__(self, vendor_id, product_id):
        self.device = self.find_device(vendor_id, product_id)
        if self.device:
            self.handle = self.device.open()
            self.claim_interface()

    def find_device(self, vendor_id, product_id):
        # Find USB device by vendor and product ID
        for device in usb.core.find(find_all=True):
            if device.idVendor == vendor_id and device.idProduct == product_id:
                return device
        return None

    def claim_interface(self):
        # Claim USB interface for communication
        if self.device.is_kernel_driver_active(0):
            self.handle.detach_kernel_driver(0)
        self.handle.claimInterface(0)

    def send_command(self, command):
        # Send command to USB device
        self.handle.ctrlTransfer(
            bmRequestType=0x21,  # Host to device
            bRequest=0x09,       # Send command
            wValue=0,
            wIndex=0,
            data_or_wLength=command.encode(),
            timeout=1000
        )

    def read_data(self, size=64):
        # Read data from USB device
        return self.handle.interruptRead(0x81, size, timeout=1000)
```

### Ethernet
- **Bandwidth**: 10 Mbps - 10 Gbps
- **Use Case**: High-bandwidth sensors, distributed systems
- **Pros**: Long distances, reliable, deterministic (with proper config)
- **Cons**: Requires infrastructure, more complex setup

```python
# Ethernet socket communication for sensor data
import socket
import threading
import struct

class EthernetSensor:
    def __init__(self, ip_address, port=8888):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connected = False
        self.data_callback = None

    def connect(self):
        """Connect to Ethernet sensor"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.ip_address, self.port))
            self.connected = True

            # Start receiving thread
            self.receive_thread = threading.Thread(target=self.receive_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()

            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def receive_data(self):
        """Continuously receive sensor data"""
        while self.connected:
            try:
                # Receive data header (4 bytes for message size)
                header = self.socket.recv(4)
                if len(header) < 4:
                    break

                message_size = struct.unpack('!I', header)[0]

                # Receive message data
                data = b''
                while len(data) < message_size:
                    chunk = self.socket.recv(message_size - len(data))
                    if not chunk:
                        break
                    data += chunk

                # Parse sensor data
                sensor_data = self.parse_sensor_data(data)

                # Call callback if registered
                if self.data_callback:
                    self.data_callback(sensor_data)

            except Exception as e:
                print(f"Receive error: {e}")
                break

        self.connected = False

    def parse_sensor_data(self, data):
        """Parse binary sensor data"""
        # Example format: timestamp(8) + sensor_count(4) + sensor_data
        timestamp = struct.unpack('!d', data[0:8])[0]
        sensor_count = struct.unpack('!I', data[8:12])[0]

        sensors = []
        offset = 12
        for i in range(sensor_count):
            sensor_id = struct.unpack('!I', data[offset:offset+4])[0]
            value = struct.unpack('!f', data[offset+4:offset+8])[0]
            sensors.append({'id': sensor_id, 'value': value})
            offset += 8

        return {
            'timestamp': timestamp,
            'sensors': sensors
        }

    def send_command(self, command):
        """Send command to sensor"""
        if self.connected:
            try:
                self.socket.send(command.encode())
                return True
            except Exception as e:
                print(f"Send error: {e}")
        return False
```

### CAN Bus
- **Bandwidth**: Up to 1 Mbps (CAN FD up to 8 Mbps)
- **Use Case**: Automotive, industrial control, safety-critical systems
- **Pros**: Robust, deterministic, multi-master, error detection
- **Cons**: Complex setup, limited bandwidth

```python
# CAN bus communication for robotics
import can
import time

class CANInterface:
    def __init__(self, interface='socketcan', channel='can0', bitrate=500000):
        self.bus = can.interface.Bus(
            interface=interface,
            channel=channel,
            bitrate=bitrate
        )
        self.message_handlers = {}

        # Start receiving thread
        self.receive_thread = threading.Thread(target=self.receive_messages)
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def register_handler(self, arbitration_id, handler):
        """Register message handler for specific arbitration ID"""
        self.message_handlers[arbitration_id] = handler

    def receive_messages(self):
        """Continuously receive CAN messages"""
        while True:
            try:
                message = self.bus.recv(timeout=1.0)
                if message:
                    self.handle_message(message)
            except Exception as e:
                print(f"CAN receive error: {e}")

    def handle_message(self, message):
        """Handle received CAN message"""
        arbitration_id = message.arbitration_id

        if arbitration_id in self.message_handlers:
            self.message_handlers[arbitration_id](message)

    def send_message(self, arbitration_id, data, extended_id=False):
        """Send CAN message"""
        message = can.Message(
            arbitration_id=arbitration_id,
            data=data,
            extended_id=extended_id
        )

        try:
            self.bus.send(message)
            return True
        except Exception as e:
            print(f"CAN send error: {e}")
            return False

# Motor controller using CAN bus
class CANMotorController:
    def __init__(self, can_interface, motor_id):
        self.can_interface = can_interface
        self.motor_id = motor_id

        # Register message handlers
        self.can_interface.register_handler(
            0x100 + motor_id,  # Status message ID
            self.handle_status_message
        )

        self.motor_status = {
            'position': 0.0,
            'velocity': 0.0,
            'current': 0.0,
            'temperature': 0.0
        }

    def handle_status_message(self, message):
        """Handle motor status feedback"""
        if len(message.data) == 16:
            # Parse motor status (position, velocity, current, temperature)
            self.motor_status['position'] = struct.unpack('!f', message.data[0:4])[0]
            self.motor_status['velocity'] = struct.unpack('!f', message.data[4:8])[0]
            self.motor_status['current'] = struct.unpack('!f', message.data[8:12])[0]
            self.motor_status['temperature'] = struct.unpack('!f', message.data[12:16])[0]

    def set_position(self, position):
        """Command motor to move to position"""
        command_data = struct.pack('!f', position)
        return self.can_interface.send_message(0x200 + self.motor_id, command_data)

    def set_velocity(self, velocity):
        """Command motor velocity"""
        command_data = struct.pack('!f', velocity)
        return self.can_interface.send_message(0x300 + self.motor_id, command_data)

    def enable_motor(self):
        """Enable motor power"""
        return self.can_interface.send_message(0x400 + self.motor_id, b'\x01')

    def disable_motor(self):
        """Disable motor power"""
        return self.can_interface.send_message(0x400 + self.motor_id, b'\x00')
```

## 📡 Sensor Hardware

### Vision Sensors

#### USB Cameras
```python
# USB camera interface using OpenCV
import cv2
import numpy as np

class USBCamera:
    def __init__(self, camera_id=0, resolution=(640, 480), fps=30):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps

        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise Exception(f"Cannot open camera {camera_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.frame_count = 0
        self.last_time = time.time()

    def capture_frame(self):
        """Capture single frame from camera"""
        ret, frame = self.cap.read()

        if ret:
            self.frame_count += 1
            current_time = time.time()

            # Calculate actual FPS
            if current_time - self.last_time > 1.0:
                actual_fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time
                print(f"Camera FPS: {actual_fps:.1f}")

        return frame if ret else None

    def capture_continuous(self, callback):
        """Capture frames continuously with callback"""
        while True:
            frame = self.capture_frame()
            if frame is not None:
                callback(frame)

            # Small delay to prevent CPU overload
            cv2.waitKey(1)

    def set_auto_exposure(self, enable):
        """Set camera auto exposure"""
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if enable else 0)

    def set_exposure(self, exposure):
        """Set camera exposure value"""
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def release(self):
        """Release camera resources"""
        self.cap.release()
```

#### Intel RealSense Depth Cameras
```python
# Intel RealSense depth camera interface
import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Get depth sensor for setting parameters
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        print(f"Depth scale is: {depth_scale}")

        # Create alignment object
        self.align = rs.align(rs.stream.color)

    def get_frame(self):
        """Get aligned color and depth frames"""
        frames = self.pipeline.wait_for_frames()

        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return None, None

        # Convert to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def get_point_cloud(self, color_image, depth_image):
        """Convert depth image to 3D point cloud"""
        # Create point cloud object
        pc = rs.pointcloud()
        points = pc.calculate(aligned_depth_frame)

        # Get point cloud data
        vertices = np.asanyarray(points.get_vertices())
        colors = np.asanyarray(points.get_texture_coordinates())

        return vertices, colors

    def stop(self):
        """Stop camera streaming"""
        self.pipeline.stop()
```

### Distance Sensors

#### Ultrasonic Sensors
```python
# Ultrasonic sensor array
import RPi.GPIO as GPIO
import time

class UltrasonicArray:
    def __init__(self, sensor_configs):
        """
        sensor_configs: list of {'trigger': pin, 'echo': pin, 'angle': angle}
        """
        self.sensors = sensor_configs
        GPIO.setmode(GPIO.BCM)

        # Setup GPIO pins for each sensor
        for config in self.sensors:
            GPIO.setup(config['trigger'], GPIO.OUT)
            GPIO.setup(config['echo'], GPIO.IN)

    def read_distance(self, trigger_pin, echo_pin):
        """Read distance from single ultrasonic sensor"""
        # Send pulse
        GPIO.output(trigger_pin, GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(trigger_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(trigger_pin, GPIO.LOW)

        # Measure echo time
        start_time = time.time()
        stop_time = time.time()

        while GPIO.input(echo_pin) == 0:
            start_time = time.time()

        while GPIO.input(echo_pin) == 1:
            stop_time = time.time()

        # Calculate distance
        elapsed_time = stop_time - start_time
        distance = (elapsed_time * 34300) / 2

        return distance

    def scan_all_sensors(self):
        """Read all sensors in the array"""
        scan_data = []

        for i, config in enumerate(self.sensors):
            distance = self.read_distance(config['trigger'], config['echo'])
            scan_data.append({
                'sensor_id': i,
                'angle': config['angle'],
                'distance': distance,
                'timestamp': time.time()
            })

        return scan_data

    def get_obstacle_map(self, scan_data):
        """Convert sensor readings to obstacle points"""
        obstacles = []

        for reading in scan_data:
            if 0 < reading['distance'] < 400:  # Valid range 0-4m
                angle_rad = math.radians(reading['angle'])
                x = reading['distance'] * math.cos(angle_rad)
                y = reading['distance'] * math.sin(angle_rad)

                obstacles.append({
                    'x': x,
                    'y': y,
                    'sensor_id': reading['sensor_id']
                })

        return obstacles
```

#### LiDAR Sensors
```python
# RPLidar A1 interface
import serial
import math
import time

class RPLidar:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.scanning = False

    def connect(self):
        """Connect to LiDAR"""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)

            # Start motor
            self.start_motor()

            # Start scanning
            self.start_scan()

            self.scanning = True
            return True
        except Exception as e:
            print(f"LiDAR connection failed: {e}")
            return False

    def start_motor(self):
        """Start LiDAR motor"""
        self.serial.write(b'\xA5\x25')

    def stop_motor(self):
        """Stop LiDAR motor"""
        self.serial.write(b'\xA5\x25')

    def start_scan(self):
        """Start continuous scanning"""
        self.serial.write(b'\xA5\x20')

    def stop_scan(self):
        """Stop scanning"""
        self.serial.write(b'\xA5\x25')

    def get_scan_data(self):
        """Get complete 360-degree scan"""
        if not self.scanning:
            return []

        scan_points = []
        scan_start_time = time.time()

        # Read data for one complete rotation
        while time.time() - scan_start_time < 0.1:  # 10Hz scan rate
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting)

                # Parse RPLidar data format
                for i in range(0, len(data), 5):
                    if i + 5 <= len(data):
                        # Check for valid data packet
                        if data[i] == 0x3A:  # Scan data start
                            quality = data[i + 1]
                            angle = (data[i + 3] << 8) | data[i + 2]
                            distance = (data[i + 5] << 8) | data[i + 4]

                            # Convert to degrees and millimeters
                            angle_deg = angle / 64.0
                            distance_mm = distance

                            if quality > 0 and distance_mm > 0:
                                scan_points.append({
                                    'angle': angle_deg,
                                    'distance': distance_mm,
                                    'quality': quality
                                })

        return scan_points

    def get_cartesian_points(self, scan_points):
        """Convert polar scan points to Cartesian coordinates"""
        cartesian_points = []

        for point in scan_points:
            angle_rad = math.radians(point['angle'])
            distance_m = point['distance'] / 1000.0  # Convert to meters

            x = distance_m * math.cos(angle_rad)
            y = distance_m * math.sin(angle_rad)

            cartesian_points.append({
                'x': x,
                'y': y,
                'angle': point['angle'],
                'distance': point['distance'],
                'quality': point['quality']
            })

        return cartesian_points

    def disconnect(self):
        """Disconnect from LiDAR"""
        self.scanning = False
        if self.serial:
            self.stop_scan()
            self.stop_motor()
            self.serial.close()
```

## ⚙️ Actuator Hardware

### DC Motors with Encoders
```python
# DC motor controller with encoder feedback
import RPi.GPIO as GPIO
import time
import threading

class DCMotor:
    def __init__(self, pwm_pin, dir1_pin, dir2_pin, encoder_a_pin, encoder_b_pin):
        # Motor control pins
        self.pwm_pin = pwm_pin
        self.dir1_pin = dir1_pin
        self.dir2_pin = dir2_pin

        # Encoder pins
        self.encoder_a_pin = encoder_a_pin
        self.encoder_b_pin = encoder_b_pin

        # Motor state
        self.speed = 0
        self.position = 0
        self.velocity = 0
        self.last_encoder_time = time.time()
        self.last_encoder_count = 0

        # Setup GPIO
        GPIO.setup(pwm_pin, GPIO.OUT)
        GPIO.setup(dir1_pin, GPIO.OUT)
        GPIO.setup(dir2_pin, GPIO.OUT)
        GPIO.setup(encoder_a_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(encoder_b_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Setup PWM
        self.pwm = GPIO.PWM(pwm_pin, 1000)  # 1kHz frequency
        self.pwm.start(0)

        # Setup encoder interrupts
        GPIO.add_event_detect(encoder_a_pin, GPIO.BOTH, callback=self.encoder_callback)

        # Control thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def encoder_callback(self, channel):
        """Handle encoder interrupts"""
        # Read encoder B to determine direction
        b_state = GPIO.input(self.encoder_b_pin)

        # Update position based on direction
        if b_state == GPIO.HIGH:
            self.position += 1
        else:
            self.position -= 1

        # Calculate velocity
        current_time = time.time()
        dt = current_time - self.last_encoder_time

        if dt > 0:
            self.velocity = (self.position - self.last_encoder_count) / dt
            self.last_encoder_count = self.position
            self.last_encoder_time = current_time

    def set_speed(self, speed):
        """Set motor speed (-100 to 100)"""
        self.speed = max(-100, min(100, speed))

    def control_loop(self):
        """Motor control loop"""
        while True:
            # Set direction based on speed
            if self.speed > 0:
                GPIO.output(self.dir1_pin, GPIO.HIGH)
                GPIO.output(self.dir2_pin, GPIO.LOW)
            elif self.speed < 0:
                GPIO.output(self.dir1_pin, GPIO.LOW)
                GPIO.output(self.dir2_pin, GPIO.HIGH)
            else:
                GPIO.output(self.dir1_pin, GPIO.LOW)
                GPIO.output(self.dir2_pin, GPIO.LOW)

            # Set PWM duty cycle
            self.pwm.ChangeDutyCycle(abs(self.speed))

            time.sleep(0.01)  # 100Hz control frequency

    def get_position(self):
        """Get current motor position (encoder counts)"""
        return self.position

    def get_velocity(self):
        """Get current motor velocity (counts/second)"""
        return self.velocity

    def reset_position(self):
        """Reset position to zero"""
        self.position = 0
        self.last_encoder_count = 0
```

### Servo Motors
```python
# Servo motor controller
import RPi.GPIO as GPIO
import time

class Servo:
    def __init__(self, pwm_pin, min_pulse=0.5, max_pulse=2.5, min_angle=0, max_angle=180):
        self.pwm_pin = pwm_pin
        self.min_pulse = min_pulse  # ms
        self.max_pulse = max_pulse  # ms
        self.min_angle = min_angle
        self.max_angle = max_angle

        # Setup PWM for servo (50Hz = 20ms period)
        GPIO.setup(pwm_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(pwm_pin, 50)
        self.pwm.start(0)

        self.current_angle = None

    def set_angle(self, angle):
        """Set servo angle"""
        # Clamp angle to valid range
        angle = max(self.min_angle, min(self.max_angle, angle))

        # Convert angle to pulse width
        pulse_range = self.max_pulse - self.min_pulse
        angle_range = self.max_angle - self.min_angle

        pulse_width = self.min_pulse + (angle - self.min_angle) * pulse_range / angle_range

        # Convert to duty cycle (0-100%)
        duty_cycle = pulse_width * 100 / 20  # 20ms period

        self.pwm.ChangeDutyCycle(duty_cycle)
        self.current_angle = angle

        # Short delay for servo to reach position
        time.sleep(0.1)
        self.pwm.ChangeDutyCycle(0)  # Reduce jitter

    def get_angle(self):
        """Get current servo angle"""
        return self.current_angle

# Multi-servo controller
class RobotArm:
    def __init__(self, servo_configs):
        """
        servo_configs: list of {'pin': int, 'name': str, 'min_angle': float, 'max_angle': float}
        """
        self.servos = {}

        for config in servo_configs:
            servo = Servo(
                config['pin'],
                min_angle=config.get('min_angle', 0),
                max_angle=config.get('max_angle', 180)
            )
            self.servos[config['name']] = servo

    def set_servo_angle(self, servo_name, angle):
        """Set angle for specific servo"""
        if servo_name in self.servos:
            self.servos[servo_name].set_angle(angle)
            return True
        return False

    def set_arm_position(self, positions):
        """Set all servo positions"""
        for servo_name, angle in positions.items():
            self.set_servo_angle(servo_name, angle)

    def get_arm_position(self):
        """Get current arm position"""
        position = {}
        for name, servo in self.servos.items():
            position[name] = servo.get_angle()
        return position
```

## 🔧 Hardware Integration with ROS 2

### ROS 2 Hardware Drivers
```python
# ROS 2 camera driver
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class ROS2CameraNode(Node):
    def __init__(self, camera_id=0):
        super().__init__('ros2_camera')

        self.camera_id = camera_id
        self.bridge = CvBridge()

        # Publishers
        self.image_publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.info_publisher = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        # Camera parameters
        self.declare_parameter('camera_id', camera_id)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)

        # Initialize camera
        self.camera = USBCamera(
            camera_id=self.get_parameter('camera_id').value,
            resolution=(self.get_parameter('width').value, self.get_parameter('height').value),
            fps=self.get_parameter('fps').value
        )

        # Camera info
        self.camera_info = self.create_camera_info()

        # Timer for publishing
        timer_period = 1.0 / self.get_parameter('fps').value
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('ROS 2 Camera Node started')

    def create_camera_info(self):
        """Create camera info message"""
        info = CameraInfo()
        info.header.frame_id = self.get_parameter('frame_id').value

        # Camera parameters (would be calibrated in real system)
        info.width = self.get_parameter('width').value
        info.height = self.get_parameter('height').value
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Camera matrix (intrinsic parameters)
        info.k = [
            500.0, 0.0, info.width/2,
            0.0, 500.0, info.height/2,
            0.0, 0.0, 1.0
        ]

        # Rectification matrix
        info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        # Projection matrix
        info.p = [
            500.0, 0.0, info.width/2, 0.0,
            0.0, 500.0, info.height/2, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        return info

    def timer_callback(self):
        """Timer callback for publishing camera data"""
        frame = self.camera.capture_frame()

        if frame is not None:
            # Convert OpenCV image to ROS message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = self.get_parameter('frame_id').value

            # Publish image
            self.image_publisher.publish(ros_image)

            # Publish camera info
            self.camera_info.header.stamp = ros_image.header.stamp
            self.info_publisher.publish(self.camera_info)

    def destroy_node(self):
        """Clean up resources"""
        self.camera.release()
        super().destroy_node()
```

### Hardware Abstraction Layer
```python
# Hardware abstraction layer for robotics
class HardwareManager:
    def __init__(self):
        self.devices = {}
        self.device_types = {
            'motor': DCMotor,
            'servo': Servo,
            'camera': USBCamera,
            'lidar': RPLidar,
            'ultrasonic': UltrasonicArray
        }

    def add_device(self, name, device_type, config):
        """Add hardware device to manager"""
        if device_type in self.device_types:
            try:
                device = self.device_types[device_type](**config)
                self.devices[name] = device
                return True
            except Exception as e:
                print(f"Failed to add device {name}: {e}")
                return False
        else:
            print(f"Unknown device type: {device_type}")
            return False

    def get_device(self, name):
        """Get device by name"""
        return self.devices.get(name)

    def remove_device(self, name):
        """Remove device from manager"""
        if name in self.devices:
            device = self.devices[name]
            # Cleanup device if needed
            if hasattr(device, 'cleanup'):
                device.cleanup()
            del self.devices[name]
            return True
        return False

    def list_devices(self):
        """List all managed devices"""
        device_list = []
        for name, device in self.devices.items():
            device_list.append({
                'name': name,
                'type': type(device).__name__
            })
        return device_list

    def update_all(self):
        """Update all devices (for real-time control)"""
        for device in self.devices.values():
            if hasattr(device, 'update'):
                device.update()
```

---

## 🎯 Best Practices

### Hardware Selection

1. **Computational Requirements**: Match processing power to AI and control needs
2. **Real-time Capabilities**: Use appropriate hardware for time-critical operations
3. **Power Consumption**: Consider battery life and thermal management
4. **Interface Compatibility**: Ensure communication interfaces match requirements
5. **Environmental Conditions**: Consider temperature, vibration, and humidity

### Integration Guidelines

1. **Modular Design**: Separate hardware interfaces from application logic
2. **Error Handling**: Implement robust error detection and recovery
3. **Testing**: Test hardware interfaces thoroughly before deployment
4. **Documentation**: Maintain detailed hardware interface documentation
5. **Safety**: Implement safety mechanisms for actuator control

### Performance Optimization

1. **Buffer Management**: Use appropriate buffer sizes for sensor data
2. **Thread Safety**: Protect shared hardware resources with proper synchronization
3. **Priority Scheduling**: Use real-time priorities for critical hardware control
4. **Memory Management**: Avoid memory leaks in long-running hardware drivers
5. **Interrupt Handling**: Minimize interrupt latency for time-critical operations

---

## 🎉 Chapter Summary

Hardware is the physical foundation of robotics systems, providing the sensing, computation, and actuation capabilities that bring robots to life:

1. **Computing Platforms**: Range from microcontrollers for real-time control to powerful GPUs for AI processing
2. **Communication Interfaces**: USB, Ethernet, CAN bus, and other protocols connect hardware components
3. **Sensors**: Vision, distance, and other sensors provide environmental awareness
4. **Actuators**: Motors and servos enable physical interaction with the world
5. **ROS 2 Integration**: Hardware abstraction layers connect physical devices to software systems

Understanding hardware capabilities and limitations is essential for designing effective robotic systems that can operate reliably in the real world.

**[← Back to Chapter 4: Distributed Systems](04-distributed-systems.md) | [Quarter 1 Complete: Review and Next Steps →](../quarter-2/index.md)**

## Chapter 5 Knowledge Check

### Question 1: Which computing platform is best for AI-powered robots requiring GPU acceleration?

**Options:**
- A) Raspberry Pi
- B) Arduino
- C) NVIDIA Jetson
- D) Intel NUC

**Answer**
> **Correct Answer:** C) NVIDIA Jetson
>
> NVIDIA Jetson platforms provide GPU acceleration with CUDA support, making them ideal for AI-powered robots requiring computer vision, deep learning, and parallel processing capabilities that other platforms lack.

---

### Question 2: What communication interface is commonly used for automotive and safety-critical systems?

**Options:**
- A) USB
- B) Ethernet
- C) CAN Bus
- D) WiFi

**Answer**
> **Correct Answer:** C) CAN Bus
>
> CAN Bus is the standard communication interface for automotive and safety-critical systems due to its robust error detection, deterministic behavior, multi-master capability, and reliability in harsh electrical environments.

---

### Question 3: Which sensor type provides 3D point cloud data for robotics navigation?

**Options:**
- A) USB Camera
- B) Intel RealSense
- C) Ultrasonic Sensor
- D) LiDAR

**Answer**
> **Correct Answer:** D) LiDAR
>
> LiDAR sensors provide 3D point cloud data by measuring distance using laser pulses, offering 360-degree environmental scanning with high precision and range, making them essential for robotics navigation and mapping.

---

### Question 4: What is the main advantage of using microcontrollers for motor control?

**Options:**
- A) High processing power
- B) Large memory capacity
- C) Real-time performance
- D) Easy programming

**Answer**
> **Correct Answer:** C) Real-time performance
>
> Microcontrollers provide deterministic, real-time performance essential for motor control, with minimal latency, precise timing control, and the ability to respond immediately to sensor feedback and control commands.

---

### Question 5: Which component is essential for bridging physical hardware with ROS 2 software?

**Options:**
- A) Simulation environment
- B) Hardware abstraction layer
- C) Web interface
- D) Cloud service

**Answer**
> **Correct Answer:** B) Hardware abstraction layer
>
> A hardware abstraction layer (HAL) provides a standardized interface between physical hardware devices and ROS 2 software, enabling seamless integration, device management, and communication while hiding hardware-specific implementation details.