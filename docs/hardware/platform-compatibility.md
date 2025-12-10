---
title: "Platform Compatibility Guide"
sidebar_label: "Platform Compatibility"
sidebar_position: 8
---

# Platform Compatibility Guide

## Supported Operating Systems and Hardware Configurations

This comprehensive guide details which platforms are officially supported for humanoid robotics development with ROS 2 and provides compatibility information for different hardware architectures.

## 🖥️ Operating System Compatibility

### ✅ **Recommended: Ubuntu Linux**

| Ubuntu Version | ROS 2 Support | Status | Notes |
|----------------|---------------|--------|-------|
| **22.04 LTS** | Humble, Iron, Rolling | ✅ **Fully Supported** | Primary development platform |
| 20.04 LTS | Foxy, Galactic | ✅ **Supported** | LTS, stable, widely used |
| 23.10 | Humble, Iron, Rolling | ✅ **Supported** | Latest stable release |
| 24.04 LTS | Humble, Iron, Rolling | ⚠️ **Tested** | Newest LTS (verify compatibility) |

**Why Ubuntu is Recommended:**
- Native ROS 2 support and regular updates
- Largest community support base
- Professional development tools availability
- Most robotics software packages target Ubuntu first

---

### ✅ **Alternative: Debian**

| Debian Version | ROS 2 Support | Status | Notes |
|----------------|---------------|--------|-------|
| 12 (Bookworm) | Humble, Iron, Rolling | ✅ **Supported** | Stable, similar to Ubuntu |
| 11 (Bullseye) | Foxy, Galactic | ✅ **Supported** | Well-established, reliable |
| Testing | Rolling | ⚠️ **Experimental** | May work but not guaranteed |

---

### ✅ **Windows with WSL2**

| Windows Version | WSL2 Support | ROS 2 Status | Setup Complexity |
|-----------------|---------------|--------------|------------------|
| **Windows 11** | ✅ Native | ✅ **Fully Supported** | Easy (one-click install) |
| **Windows 10** | ✅ Available | ✅ **Supported** | Moderate (manual setup) |
| Windows Server 2022 | ✅ Available | ✅ **Supported** | Moderate |

**WSL2 Benefits:**
- Native Linux kernel within Windows
- Full ROS 2 compatibility
- Windows and Linux tools integration
- GPU support for ML/simulation

**WSL2 Setup Requirements:**
- Windows 10 version 2004 or higher
- Virtual Machine Platform feature enabled
- WSL2 Linux distribution (Ubuntu recommended)

---

### ✅ **macOS**

| macOS Version | Method | ROS 2 Support | Status |
|---------------|--------|---------------|--------|
| **13 (Ventura)** | Docker/VirtualBox | ✅ **Supported** | Recommended setup |
| 12 (Monterey) | Docker/VirtualBox | ✅ **Supported** | Good compatibility |
| 11 (Big Sur) | Docker/VirtualBox | ✅ **Supported** | Works well |
| Intel-based Macs | Bootcamp | ✅ **Supported** | Native Ubuntu install |

**macOS Considerations:**
- No native ROS 2 support (requires virtualization)
- Docker-based development works excellently
- M1/M2 Macs need specific Docker configuration
- Virtualization performance is excellent on Apple Silicon

---

### ⚠️ **Limited Support**

| Platform | Status | Limitations |
|----------|--------|-------------|
| **Arch Linux** | ⚠️ **Community** | AUR packages available, no official support |
| **Fedora** | ⚠️ **Community** | Community packages, may require manual setup |
| **openSUSE** | ❌ **Not Recommended** | Very limited package availability |
| **CentOS/RHEL** | ❌ **Not Supported** | Package availability issues |
| **FreeBSD** | ❌ **Not Supported** | No ROS 2 support |

---

## 🏗️ Hardware Architecture Compatibility

### ✅ **x86_64 (Intel/AMD 64-bit)**

**Processor Support:**
- ✅ Intel Core i3/i5/i7/i9 (6th gen or newer)
- ✅ AMD Ryzen 3/5/7/9 (all generations)
- ✅ Intel Xeon (all modern generations)
- ✅ AMD EPYC (server/workstation)

**Recommended Specifications:**
- **Minimum:** 2 cores, 4GB RAM, 20GB storage
- **Recommended:** 4+ cores, 8GB+ RAM, 50GB+ SSD
- **Advanced:** 8+ cores, 16GB+ RAM, 100GB+ NVMe SSD

---

### ✅ **ARM64 (AArch64)**

**Processor Support:**
- ✅ **Raspberry Pi 4** (8GB model recommended)
- ✅ **NVIDIA Jetson Series** (Nano, Xavier, Orin)
- ✅ **Apple M1/M2/M3** (via Docker/VM)
- ✅ **AWS Graviton** (cloud instances)

**Performance Tiers:**
- **Basic:** Raspberry Pi 4 (4GB) - Light development, simulation
- **Intermediate:** Raspberry Pi 4 (8GB), Jetson Nano - Full development
- **Advanced:** Jetson Xavier/Orin - AI/ML, computer vision
- **Professional:** Custom ARM servers - Large-scale deployments

---

### ⚠️ **Limited/Experimental**

| Architecture | Status | Platforms | Notes |
|--------------|--------|-----------|-------|
| **ARM32** | ⚠️ **Limited** | Raspberry Pi 3, older boards | Some ROS 2 packages missing |
| **RISC-V** | ❌ **Experimental** | Development boards | No official ROS 2 support |
| **PowerPC** | ❌ **Not Supported** | Legacy systems | Modern ROS 2 doesn't support |
| **Itanium** | ❌ **Not Supported** | Enterprise servers | Deprecated architecture |

---

## 📱 Device Compatibility Matrix

### **Computer Hardware**

| Component | Minimum | Recommended | Compatible | Notes |
|-----------|----------|-------------|------------|-------|
| **CPU** | Intel i3 / AMD Ryzen 3 | Intel i5+ / AMD Ryzen 5+ | ✅ | Look for 2018+ models |
| **RAM** | 4GB DDR4 | 16GB DDR4/DDR5 | ✅ | More RAM = smoother simulation |
| **Storage** | 20GB HDD | 50GB+ SSD | ✅ | SSD highly recommended |
| **Graphics** | Integrated 2GB | Dedicated 4GB+ | ✅ | NVIDIA recommended for CUDA |
| **USB** | USB 2.0 x2 | USB 3.0+ x4 | ✅ | Essential for robot hardware |
| **Network** | WiFi / 100Mbps Ethernet | WiFi 6 / Gigabit Ethernet | ✅ | Wired preferred for development |

### **Robot Controllers**

| Platform | CPU | RAM | Storage | ROS 2 | Status |
|----------|-----|-----|---------|-------|--------|
| **Raspberry Pi 4** | Cortex-A72 | 1-8GB | MicroSD | ✅ | Excellent for mobile robots |
| **NVIDIA Jetson Nano** | Cortex-A57 + Maxwell | 4-8GB | eMMC + SD | ✅ | AI/ML acceleration |
| **Intel NUC** | Core i3/i5 | 8-32GB | M.2 SSD | ✅ | Desktop performance in small form |
| **BeagleBone Black** | Cortex-A8 | 512MB | eMMC | ⚠️ **Limited** | Older, slower but functional |
| **Arduino** | AVR | 2KB | Flash | ❌ **Controller Only** | Use with ROS 2 via serial |

### **Sensors and Actuators**

| Device Type | Interface | ROS 2 Support | Compatibility | Notes |
|-------------|----------|---------------|--------------|-------|
| **USB Camera** | USB 2.0/3.0 | ✅ | ✅ | UVC compatible cameras |
| **Intel RealSense** | USB 3.0 | ✅ | ✅ | Depth cameras, 3D vision |
| **Raspberry Pi Camera** | MIPI CSI | ✅ | ✅ | Only on Raspberry Pi |
| **LiDAR (2D)** | USB/Serial/UART | ✅ | ✅ | Most Hokuyo, RPLidar models |
| **LiDAR (3D)** | Ethernet/USB | ✅ | ✅ | Velodyne, Ouster models |
| **IMU** | I2C/SPI/UART | ✅ | ✅ | MPU9250, BNO055 common |
| **Servo Motors** | PWM | ✅ | ✅ | PCA9685 controller recommended |
| **Stepper Motors** | Step/Dir | ✅ | ✅ | DRV8825, A4988 drivers |

---

## 🎮 Graphics and GPU Support

### **NVIDIA GPUs (Recommended)**

| GPU Series | CUDA Support | ROS 2 | Status | Use Cases |
|------------|-------------|-------|--------|-----------|
| **RTX 4000** | CUDA 12 | ✅ | ✅ **Excellent** | ML, computer vision |
| **RTX 3000** | CUDA 11-12 | ✅ | ✅ **Excellent** | General robotics |
| **RTX 2000** | CUDA 10-12 | ✅ | ✅ **Good** | Simulations, perception |
| **GTX 1000** | CUDA 10 | ✅ | ✅ **Good** | Basic ML, simulation |
| **GT 700** | No CUDA | ❌ | ❌ **Limited** | Only basic graphics |
| **Tesla** | CUDA | ✅ | ✅ **Professional** | Cloud, research |

### **AMD GPUs**

| GPU Series | OpenCL Support | ROS 2 | Status | Use Cases |
|------------|----------------|-------|--------|-----------|
| **RX 7000** | OpenCL 2.0+ | ✅ | ✅ **Good** | General computing |
| **RX 6000** | OpenCL 2.0+ | ✅ | ✅ **Good** | Simulations |
| **RX 5000** | OpenCL 2.0 | ✅ | ✅ **Fair** | Basic acceleration |
| **RX 4000** | OpenCL 2.0 | ✅ | ⚠️ **Limited** | Older cards |
| **Vega** | OpenCL 2.0 | ✅ | ⚠️ **Limited** | Workstation cards |

### **Integrated Graphics**

| Platform | Integrated GPU | ROS 2 | Status | Notes |
|----------|----------------|-------|--------|-------|
| **Intel HD/UHD** | Intel GPU | ✅ | ✅ **Good** | Iris Xe recommended |
| **AMD Radeon** | AMD GPU | ✅ | ✅ **Good** | Vega graphics decent |
| **Apple Silicon** | Apple GPU | ⚠️ **Limited** | ⚠️ **Docker only** | M1/M2/M3 work in Docker |
| **Raspberry Pi** | VideoCore VI | ✅ | ✅ **Basic** | 4K video decode, no acceleration |

---

## 🔧 Development Tools Compatibility

### **IDEs and Editors**

| Tool | Windows | macOS | Linux | Notes |
|------|---------|-------|-------|-------|
| **VS Code** | ✅ | ✅ | ✅ | **Recommended** - ROS 2 extensions available |
| **PyCharm** | ✅ | ✅ | ✅ | Professional Python development |
| **CLion** | ✅ | ✅ | ✅ | Professional C++ development |
| **Vim/Emacs** | ✅ | ✅ | ✅ | Terminal-based editing |
| **Jupyter Notebooks** | ✅ | ✅ | ✅ | Data analysis, ML prototyping |

### **Version Control**

| Tool | Platform Support | ROS 2 Integration |
|------|------------------|--------------------|
| **Git** | ✅ All platforms | ✅ Essential |
| **GitHub Desktop** | ✅ Windows/macOS | ✅ Good for beginners |
| **GitKraken** | ✅ Cross-platform | ✅ Professional features |
| **Sourcetree** | ✅ Windows/macOS | ✅ Free version available |

### **Container Platforms**

| Platform | Windows | macOS | Linux | ROS 2 Use |
|----------|---------|-------|-------|------------|
| **Docker** | ✅ | ✅ | ✅ | **Recommended** - Cross-platform development |
| **Podman** | ❌ | ✅ | ✅ | Alternative to Docker |
| **Singularity** | ❌ | ✅ | ✅ | HPC/Research environments |
| **LXC/LXD** | ❌ | ❌ | ✅ | Linux containerization |

---

## 🌐 Cloud Platform Compatibility

### **Development Platforms**

| Platform | ROS 2 Support | Pricing | Use Cases |
|----------|---------------|--------|-----------|
| **GitHub Codespaces** | ✅ | Pay-per-use | Development environments |
| **GitPod** | ✅ | Free tier available | Cloud IDE |
| **AWS** | ✅ | Various | Large-scale simulation |
| **Google Cloud** | ✅ | Various | AI/ML integration |
| **Azure** | ✅ | Various | Enterprise robotics |

### **Cloud GPU Services**

| Service | GPU Options | ROS 2 | Performance | Cost |
|---------|-------------|-------|-------------|------|
| **AWS EC2** | NVIDIA T4, V100, A100 | ✅ | Excellent | Premium |
| **Google Cloud AI** | NVIDIA T4, V100 | ✅ | Excellent | Premium |
| **Azure** | NVIDIA V100, A100 | ✅ | Excellent | Premium |
| **Paperspace** | NVIDIA RTX series | ✅ | Good | Mid-range |
| **Vast.ai** | Various GPUs | ✅ | Variable | Budget |

---

## 📊 Compatibility Testing Results

### **Automated Testing Status**

| Platform | Build Status | Unit Tests | Integration Tests | Documentation |
|----------|-------------|-----------|------------------|--------------|
| **Ubuntu 22.04** | ✅ | ✅ | ✅ | ✅ |
| **Windows 11 + WSL2** | ✅ | ✅ | ✅ | ✅ |
| **macOS + Docker** | ✅ | ✅ | ⚠️ | ✅ |
| **Ubuntu 20.04** | ✅ | ✅ | ✅ | ✅ |
| **Debian 12** | ✅ | ✅ | ⚠️ | ⚠️ |
| **Raspberry Pi 4** | ✅ | ⚠️ | ❌ | ✅ |

**Status Legend:**
- ✅ **Passing** - All tests pass
- ⚠️ **Partial** - Some tests pass with limitations
- ❌ **Failing** - Tests not passing or not tested

---

## 🔍 Troubleshooting Common Issues

### **Installation Problems**

| Issue | Platform | Solution |
|-------|----------|----------|
| **ROS 2 Dependencies** | All | Use `rosdep` to install dependencies |
| **Python Version** | All | Ensure Python 3.8+ is installed |
| **CMake Version** | Older systems | Upgrade to CMake 3.10+ |
| **Graphics Drivers** | NVIDIA | Install CUDA toolkit and drivers |
| **USB Permissions** | Linux | Add user to `dialout` group |

### **Runtime Issues**

| Issue | Platform | Solution |
|-------|----------|----------|
| **Can't find ROS packages** | All | Source setup.bash in terminal |
| **Simulation crashes** | All | Check graphics drivers, use software rendering |
| **Robot not responding** | All | Verify USB permissions, check dmesg logs |
| **High CPU usage** | ARM platforms | Use optimized builds, limit simulation complexity |
| **Network issues** | Cloud setups | Configure firewall, check port forwarding |

### **Performance Issues**

| Issue | Cause | Solution |
|-------|-------|----------|
| **Slow simulation** | Integrated graphics | Reduce simulation complexity, upgrade GPU |
| **High latency** | WiFi connection | Use wired Ethernet, improve signal |
| **Out of memory** | RAM limitations | Close other applications, increase swap space |
| **Throttling** | Overheating | Improve cooling, check thermal paste |
| **Storage bottlenecks** | HDD storage | Upgrade to SSD, defragment drive |

---

## 🚀 Getting Started Guide

### **Step 1: Check Current System**
```bash
# Check OS version
lsb_release -a  # Linux
system_profiler SPSoftwareDataType  # macOS
systeminfo | findstr /B "OS Name"  # Windows

# Check architecture
uname -m

# Check available disk space
df -h

# Check memory
free -h  # Linux
vm_stat  # macOS
```

### **Step 2: Verify Requirements**
```bash
# Check Python version
python3 --version

# Check CMake version
cmake --version

# Check Git version
git --version
```

### **Step 3: Prepare System**
```bash
# Update packages (Ubuntu/Debian)
sudo apt update && sudo apt upgrade -y

# Install ROS 2 dependencies
sudo apt install -y curl gnupg lsb-release
```

### **Step 4: Install ROS 2**
```bash
# Follow platform-specific installation guide
# - Ubuntu: native installation
# - Windows: WSL2 setup
# - macOS: Docker installation
```

### **Step 5: Verify Installation**
```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Check ROS 2 installation
ros2 doctor

# Test basic functionality
ros2 run demo_nodes_cpp talker
```

---

## 📞 Support and Resources

### **Official Documentation**
- **ROS 2 Documentation:** https://docs.ros.org/en/humble/
- **Ubuntu Documentation:** https://help.ubuntu.com/
- **Windows WSL2:** https://learn.microsoft.com/en-us/windows/wsl/

### **Community Support**
- **ROS Discourse:** https://discourse.ros.org/
- **Stack Overflow:** https://stackoverflow.com/questions/tagged/ros2
- **GitHub Issues:** Repository-specific issues

### **Platform-Specific Help**
- **Linux:** Ubuntu Forums, Ask Ubuntu
- **Windows:** Microsoft Q&A, Super User
- **macOS:** Apple Communities, macOS Stack Exchange
- **Hardware:** Manufacturer support sites

### **Professional Support**
- **ROS Industrial:** Commercial support packages
- **Canonical:** Ubuntu LTS support
- **Microsoft:** Windows/WSL2 enterprise support
- **Docker:** Docker Enterprise support

---

**Remember:** The platform compatibility landscape is constantly evolving. This guide is regularly updated, but always check the latest ROS 2 documentation for the most current compatibility information! 🚀