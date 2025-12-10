---
title: "Platform-Specific Installation Guides"
sidebar_label: "Installation Guides"
sidebar_position: 9
---

import HardwareSpec from '@site/src/components/HardwareSpec';

# Platform-Specific Installation Guides

## Complete ROS 2 Setup for All Platforms

This guide provides step-by-step installation instructions for ROS 2 on all supported platforms. Follow the instructions for your specific operating system and hardware configuration.

## 🖥️ Ubuntu Linux (Native) - Recommended

### System Requirements

<HardwareSpec
  title="Ubuntu Native Setup Requirements"
  description="Minimum requirements for ROS 2 on Ubuntu Linux"
  costRange="Free (if already installed)"
  difficulty="Beginner"
  timeToStart="1-2 hours"
  specs={[
    { label: "OS Version", value: "Ubuntu 22.04 LTS (recommended)" },
    { label: "Architecture", value: "x86_64 (Intel/AMD 64-bit)" },
    { label: "RAM", value: "4GB minimum, 8GB recommended" },
    { label: "Storage", value: "20GB available space" },
    { label: "Network", value: "Internet connection for packages" }
  ]}
  included={[
    "Native ROS 2 performance",
    "Full hardware support",
    "Complete package availability",
    "Best community support",
    "Direct hardware access"
  ]}
  notIncluded={[
    "Windows application compatibility",
    "macOS-specific optimizations"
  ]}
/>

### Installation Steps

#### Step 1: System Preparation
```bash
# Update package index
sudo apt update

# Upgrade existing packages
sudo apt upgrade -y

# Install required tools
sudo apt install -y curl gnupg lsb-release
```

#### Step 2: Add ROS 2 Repository
```bash
# Add ROS 2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Add ROS 2 repository
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Update package index
sudo apt update
```

#### Step 3: Install ROS 2
```bash
# Install full desktop version (recommended)
sudo apt install -y ros-humble-desktop

# Or install minimal version
# sudo apt install -y ros-humble-base
```

#### Step 4: Install Development Tools
```bash
# Install ROS 2 development tools
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Install additional tools
sudo apt install -y python3-colcon-common-extensions python3-pip
```

#### Step 5: Initialize rosdep
```bash
sudo rosdep init
rosdep update
```

#### Step 6: Setup Environment
```bash
# Add to ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### Step 7: Verify Installation
```bash
# Check ROS 2 installation
ros2 doctor

# Test basic functionality
ros2 run demo_nodes_cpp talker
```

---

## 🪟 Windows with WSL2

### System Requirements

<HardwareSpec
  title="Windows WSL2 Setup Requirements"
  description="Requirements for ROS 2 on Windows using WSL2"
  costRange="Free (built into Windows)"
  difficulty="Intermediate"
  timeToStart="2-3 hours"
  specs={[
    { label: "Windows Version", value: "Windows 10 version 2004+ or Windows 11" },
    { label: "RAM", value: "8GB minimum, 16GB recommended" },
    { label: "Storage", value: "40GB available space" },
    { label: "Virtualization", value: "Enabled in BIOS/UEFI" },
    { label: "Architecture", value: "x86_64 only" }
  ]}
  included={[
    "Windows and Linux integration",
    "Access to Windows tools",
    "Good performance",
    "Easy setup process"
  ]}
  notIncluded={[
    "Direct hardware access (limited)",
    "Native GPU acceleration for some applications"
  ]}
/>

### Installation Steps

#### Step 1: Enable WSL2 and Virtual Machine Platform
```powershell
# Run PowerShell as Administrator
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart your computer
```

#### Step 2: Set WSL2 as Default
```powershell
# Set WSL2 as default
wsl --set-default-version 2
```

#### Step 3: Install Ubuntu from Microsoft Store
1. Open Microsoft Store
2. Search for "Ubuntu 22.04 LTS"
3. Click "Install" (or "Get")
4. Launch Ubuntu from Start Menu
5. Create username and password

#### Step 4: Install ROS 2 in WSL2
Follow the Ubuntu installation steps above inside your WSL2 Ubuntu terminal.

#### Step 5: Configure X11 Forwarding (Optional)
```bash
# Install X11 server on Windows (e.g., VcXsrv or X410)
# Then configure in WSL2
echo "export DISPLAY=:0" >> ~/.bashrc
```

#### Step 6: Verify Installation
```bash
# Test ROS 2 commands
ros2 doctor
ros2 run demo_nodes_cpp talker
```

---

## 🍎 macOS with Docker

### System Requirements

<HardwareSpec
  title="macOS Docker Setup Requirements"
  description="Requirements for ROS 2 on macOS using Docker"
  costRange="Free for Docker Desktop"
  difficulty="Intermediate"
  timeToStart="30-60 minutes"
  specs={[
    { label: "macOS Version", value: "macOS 11 (Big Sur) or later" },
    { label: "RAM", value: "8GB minimum, 16GB recommended" },
    { label: "Storage", value:"10GB for Docker + container" },
    { label: "Processor", value: "Intel or Apple Silicon (M1/M2/M3)" },
    { label: "Virtualization", value: "Enabled by default" }
  ]}
  included={[
    "Isolated development environment",
    "Easy reproduction of setups",
    "Cross-platform compatibility",
    "No system modifications"
  ]}
  notIncluded={[
    "Native performance",
    "Direct hardware access",
    "GPU acceleration (limited on M1/M2)"
  ]}
/>

### Installation Steps

#### Step 1: Install Docker Desktop
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install and restart Mac
3. Start Docker Desktop

#### Step 2: Pull ROS 2 Docker Image
```bash
# Pull official ROS 2 image
docker pull osrf/ros:humble-desktop

# Or pull specific architecture image for Apple Silicon
# docker pull --platform linux/amd64 osrf/ros:humble-desktop
```

#### Step 3: Create Docker Compose File
```yaml
# docker-compose.yml
version: '3.8'
services:
  ros2:
    image: osrf/ros:humble-desktop
    container_name: ros2_dev
    stdin_open: true
    tty: true
    volumes:
      - ./ros2_ws:/ros2_ws
    network_mode: host
    environment:
      - DISPLAY=${DISPLAY}
    command: bash
```

#### Step 4: Run ROS 2 Container
```bash
# Start container
docker-compose up -d

# Enter container
docker-compose exec ros2 bash

# Test ROS 2
ros2 run demo_nodes_cpp talker
```

#### Step 5: Setup Development Workspace
```bash
# Inside container
cd /ros2_ws
mkdir -p src
```

---

## 🐧 Raspberry Pi 4

### System Requirements

<HardwareSpec
  title="Raspberry Pi 4 Setup Requirements"
  description="Requirements for ROS 2 on Raspberry Pi 4"
  costRange="$75-200"
  difficulty="Intermediate"
  timeToStart="2-4 hours"
  specs={[
    { label: "Model", value: "Raspberry Pi 4 (8GB recommended)" },
    { label: "Storage", value: "64GB+ high-endurance microSD" },
    { label: "Power", value: "3A+ USB-C power supply" },
    { label: "Cooling", value: "Recommended for sustained use" },
    { label: "Network", value: "Ethernet or WiFi" }
  ]}
  included={[
    "Low power consumption",
    "Embedded robotics applications",
    "GPIO access for sensors",
    "Portable development"
  ]}
  notIncluded={[
    "High-performance computing",
    "Complex simulations (limited)",
    "GPU acceleration for ML"
  ]}
/>

### Installation Steps

#### Step 1: Flash Ubuntu to microSD
```bash
# Download Ubuntu Server 22.04 for Raspberry Pi
# Flash using Raspberry Pi Imager or balenaEtcher
```

#### Step 2: Initial Setup
```bash
# SSH into Raspberry Pi (default user: ubuntu)
ssh ubuntu@<raspberry-pi-ip>

# Update system
sudo apt update && sudo apt upgrade -y
```

#### Step 3: Configure Swap (Recommended)
```bash
# Create swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Add to fstab
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### Step 4: Install ROS 2
```bash
# Follow Ubuntu installation steps
# Note: This will take longer on Raspberry Pi

# Add ROS 2 repository
sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS 2 (this may take 1-2 hours)
sudo apt update
sudo apt install -y ros-humble-desktop
```

#### Step 5: Setup Environment
```bash
# Configure ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## ☁️ Cloud Development (GitHub Codespaces)

### System Requirements

<HardwareSpec
  title="Cloud Development Setup"
  description="Requirements for ROS 2 cloud development"
  costRange="$0-50/month"
  difficulty="Beginner"
  timeToStart="5-10 minutes"
  specs={[
    { label: "Device", value: "Any computer with web browser" },
    { label: "Internet", value: "5Mbps+ connection" },
    { label: "Account", value: "GitHub account (free)" },
    { label: "Storage", value: "Cloud-based workspace" },
    { label: "Compute", value: "Cloud-hosted virtual machines" }
  ]}
  included={[
    "Instant setup",
    "No local maintenance",
    "Consistent environment",
    "Access from anywhere",
    "Automatic updates"
  ]}
  notIncluded={[
    "Physical robot control",
    "Offline development",
    "Custom hardware integration"
  ]}
/>

### Setup Steps

#### Step 1: Create GitHub Repository
1. Create new repository on GitHub
2. Clone or create your robotics project

#### Step 2: Configure Dev Container
```json
// .devcontainer/devcontainer.json
{
  "name": "ROS 2 Development",
  "image": "osrf/ros:humble-desktop",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-vscode.cpptools",
        "ms-python.python",
        "ms-azuretools.vscode-docker"
      ]
    }
  },
  "forwardPorts": [8080, 11311],
  "mounts": [
    "source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind"
  ]
}
```

#### Step 3: Launch Codespace
1. Go to your repository on GitHub
2. Click "Code" → "Codespaces" → "Create codespace"
3. Wait for environment to build (2-5 minutes)

#### Step 4: Test Setup
```bash
# Verify ROS 2 installation
ros2 doctor

# Run example
ros2 run demo_nodes_cpp talker
```

---

## 🔧 Common Post-Installation Tasks

### Install Additional Tools

#### Essential Development Tools
```bash
# Install common tools
sudo apt install -y git vim nano htop tree

# Install Python development tools
pip3 install numpy matplotlib scikit-learn opencv-python

# Install ROS 2 build tools
sudo apt install -y python3-colcon-common-extensions
```

#### Install Simulation Tools
```bash
# Install Gazebo
sudo apt install -y gazebo ros-humble-gazebo-ros-pkgs

# Install RViz2
sudo apt install -y ros-humble-rviz2

# Install additional packages
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup
```

### Create Workspace Structure
```bash
# Create ROS 2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build workspace
colcon build

# Source workspace
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Setup Git Configuration
```bash
# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Setup SSH keys (optional but recommended)
ssh-keygen -t ed25519 -C "your.email@example.com"
```

---

## 🐛 Troubleshooting Guide

### Common Installation Issues

#### Package Not Found Errors
```bash
# Update package lists
sudo apt update

# Check ROS 2 repository
grep -r 'packages.ros.org' /etc/apt/sources.list.d/

# Manually add repository if missing
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
```

#### Permission Denied Errors
```bash
# Fix ROS 2 permissions
sudo usermod -a -G dialout $USER
sudo usermod -a -G input $USER

# Log out and log back in for changes to take effect
```

#### Docker Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in
```

#### Performance Issues on Raspberry Pi
```bash
# Check CPU temperature
vcgencmd measure_temp

# Check memory usage
free -h

# Check swap usage
swapon --show
```

### Network Issues

#### Cannot Connect to ROS 2 Master
```bash
# Check ROS 2 environment
printenv | grep ROS

# Set ROS_DOMAIN_ID if needed
export ROS_DOMAIN_ID=0

# Check network connectivity
ping -c 4 8.8.8.8
```

#### Firewall Issues
```bash
# Allow ROS 2 ports
sudo ufw allow 11311
sudo ufw allow from 192.168.0.0/16
```

### Graphics Issues

#### Cannot Run RViz2 or Gazebo
```bash
# Check display variable
echo $DISPLAY

# Set display for local sessions
export DISPLAY=:0

# For Docker containers with X11
xhost +local:
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix osrf/ros:humble-desktop
```

---

## ✅ Verification Checklist

### Basic ROS 2 Setup
- [ ] ROS 2 environment sourced properly
- [ ] `ros2 doctor` shows no critical issues
- [ ] Can run talker/listener example
- [ ] Workspace builds without errors
- [ ] Can create and use ROS 2 nodes

### Advanced Features
- [ ] RViz2 launches and displays
- [ ] Gazebo simulation starts
- [ ] Camera/sensor plugins work
- [ ] Python scripts can import ROS 2
- [ ] C++ programs compile and run

### Platform-Specific
- [ ] Hardware devices accessible (Ubuntu/Windows)
- [ ] Docker containers run properly (macOS)
- [ ] GPIO access works (Raspberry Pi)
- [ ] Network connectivity stable (all platforms)

---

## 📞 Getting Help

### Official Support Channels
- **ROS 2 Documentation**: https://docs.ros.org/
- **ROS Discourse**: https://discourse.ros.org/
- **GitHub Issues**: Repository-specific

### Community Resources
- **ROS Answers**: https://answers.ros.org/
- **Stack Overflow**: Use `[ros2]` tag
- **Reddit**: r/ROS

### Platform-Specific Help
- **Ubuntu**: Ubuntu Forums, Ask Ubuntu
- **Windows**: Microsoft Q&A, Super User
- **macOS**: Apple Communities
- **Raspberry Pi**: Raspberry Pi Forums

---

**Next Steps**: Once you have ROS 2 installed and verified, proceed to [Quarter 1: The Robotic Nervous System](../quarter-1/index.md) to begin your robotics learning journey! 🚀