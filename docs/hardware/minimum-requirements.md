---
title: "Minimum Hardware Requirements"
sidebar_label: "Minimum Requirements"
sidebar_position: 6
---

import HardwareSpec from '@site/src/components/HardwareSpec';

# Minimum Hardware Requirements

## Essential Hardware for Learning Humanoid Robotics

This guide outlines the absolute minimum hardware needed to follow along with this educational book. We've designed multiple tiers to accommodate different budgets and access levels.

## 🏷️ Hardware Requirement Tiers

<HardwareSpec
  title="Minimum Requirements"
  description="Essential hardware for getting started with humanoid robotics education"
  costRange="$50-200"
  difficulty="Beginner"
  timeToStart="30 minutes"
/>

## 🎯 Quarter-by-Quarter Requirements

### Quarter 1: ROS 2 Foundations
**Minimum:** Basic computer + internet connection

| Component | Minimum | Recommended | Notes |
|-----------|----------|-------------|-------|
| **Computer** | Any modern laptop/desktop | Intel i5/AMD Ryzen 5+ | Can be Windows, macOS, or Linux |
| **RAM** | 4GB | 8GB+ | More RAM helps with complex simulations |
| **Storage** | 25GB free | 50GB+ SSD | SSD recommended for faster loading |
| **Internet** | 2Mbps | 10Mbps+ | For downloading packages and updates |

### Quarter 2: Simulation and Digital Worlds
**Minimum:** Graphics-capable computer

| Component | Minimum | Recommended | Notes |
|-----------|----------|-------------|-------|
| **CPU** | Dual-core 2.0GHz | Quad-core 3.0GHz+ | Better CPU = smoother simulations |
| **RAM** | 8GB | 16GB+ | Critical for complex Unity/Gazebo scenes |
| **Graphics** | Integrated graphics | Dedicated GPU 4GB+ | NVIDIA/AMD recommended |
| **VRAM** | 512MB | 4GB+ | Important for realistic rendering |

### Quarter 3: Perception and Sensors
**Minimum:** Computer + camera

| Component | Minimum | Recommended | Notes |
|-----------|----------|-------------|-------|
| **Camera** | USB webcam | USB 3.0/1080p | Any webcam works for basic computer vision |
| **USB Ports** | 2 USB 2.0 | 4+ USB 3.0 | For additional sensors |
| **Processor** | Same as Q2 | Same as Q2 | Computer vision benefits from better CPU |

### Quarter 4: Advanced Applications
**Minimum:** Computer + optional robot hardware

| Component | Minimum | Recommended | Notes |
|-----------|----------|-------------|-------|
| **Robot Controller** | Not required | Raspberry Pi 4+ | For physical robot implementation |
| **Sensors** | Not required | Camera + IMU | Physical sensing capabilities |
| **Motors** | Not required | Servo motor kit | Physical actuation |

---

## 🖥️ Computer Requirements

### Operating Systems

#### ✅ **Recommended: Ubuntu 22.04 LTS**
- **Why:** Native ROS 2 support, extensive community resources
- **Installation:** Can dual-boot with Windows/macOS
- **Cost:** Free
- **Performance:** Best for robotics development

#### ✅ **Alternative: Windows 10/11**
- **Why:** Familiar interface, widely available
- **Setup:** WSL2 + Ubuntu for ROS 2
- **Cost:** Often pre-installed
- **Performance:** Good with WSL2

#### ✅ **Alternative: macOS**
- **Why:** UNIX-based, familiar to developers
- **Setup:** Docker with Ubuntu or VM
- **Cost:** Pre-installed on Apple hardware
- **Performance:** Good with M1/M2 chips

### Computer Specifications

#### **Minimum Computer Specs**
```
CPU: Intel Core i3 / AMD Ryzen 3 (2018 or newer)
RAM: 8GB DDR4
Storage: 50GB free HDD space
Graphics: Intel HD Graphics 620 or newer
USB: 2x USB 3.0 ports
Network: WiFi or Ethernet
```

#### **Recommended Computer Specs**
```
CPU: Intel Core i5 / AMD Ryzen 5 (2020 or newer)
RAM: 16GB DDR4/DDR5
Storage: 100GB free SSD space
Graphics: NVIDIA GTX 1650 / AMD RX 580 or newer
USB: 4x USB 3.0/USB-C ports
Network: WiFi 6 + Ethernet
```

### Laptop vs Desktop

#### Laptops ✅
**Pros:**
- Portable, learn anywhere
- Built-in screen, keyboard, trackpad
- Battery backup
- All-in-one solution

**Cons:**
- Limited upgrade options
- Can get hot during long use
- Smaller screen size

**Best for:** Students, beginners, space-constrained setups

#### Desktops ✅
**Pros:**
- Easy to upgrade components
- Better cooling for long sessions
- Larger monitors possible
- Better price-to-performance ratio

**Cons:**
- Not portable
- Requires separate peripherals
- Takes up more space

**Best for:** Dedicated learning spaces, long-term projects

---

## 🤖 Optional Physical Robot Hardware

### **If you want to build a physical robot:**

#### **Absolute Minimum Robot Setup ($50-150)**
- **Controller:** Raspberry Pi Zero 2 W ($15)
- **Chassis:** Cardboard or 3D printed ($0-20)
- **Motors:** 2x DC gear motors ($10-20)
- **Wheels:** 3D printed or repurposed ($5-10)
- **Power:** Power bank or battery pack ($10-20)
- **Sensors:** Basic ultrasonic sensors ($5-10)

#### **Better Minimum Robot Setup ($200-400)**
- **Controller:** Raspberry Pi 4 ($75)
- **Chassis:** 3D printed robot kit ($50-100)
- **Motors:** 2x high-torque servos ($30-50)
- **Sensors:** Camera module + ultrasonic sensors ($25-40)
- **Power:** LiPo battery pack + charger ($20-30)
- **Structure:** Metal chassis ($40-80)

---

## 📱 Alternative: Cloud-Based Learning

### **No Computer Required Options**

#### **Cloud Development Environment**
- **Platforms:** GitPod, GitHub Codespaces, Replit
- **Cost:** Free tiers available
- **Setup:** Browser-based
- **Requirements:** Internet connection (5Mbps+)
- **Limitations:** No physical robot control

#### **Online Robotics Simulators**
- **Platforms:** RobotIgnite, The Construct, Gazebo Online
- **Cost:** $10-50/month
- **Setup:** Browser or remote desktop
- **Requirements:** Stable internet
- **Advantages:** Pre-configured environments

---

## 💰 Budget Breakdown by Quarter

### **Quarter 1: Foundations ($0-50)**
- Computer: $0 (use existing device)
- Internet: $0 (use existing connection)
- Software: $0 (all open-source)
- **Total:** $0

### **Quarter 2: Simulation ($0-100)**
- If current computer sufficient: $0
- Graphics upgrade (if needed): $50-100
- **Total:** $0-100

### **Quarter 3: Perception ($0-80)**
- USB webcam: $20-40
- Additional sensors: $20-40
- **Total:** $0-80

### **Quarter 4: Physical Robot ($0-400)**
- Basic robot kit: $150-300
- Advanced components: $50-100
- **Total:** $0-400

### **Overall Minimum Investment: $0-580**

---

## 🛒 Shopping Tips

### **Where to Buy Hardware**

#### **New Components**
- **Amazon:** Fast shipping, good return policy
- **SparkFun:** Quality hobbyist electronics
- **Adafruit:** Great tutorials and support
- **Digi-Key:** Professional components

#### **Used/Discounted**
- **eBay:** Save 50-70% on components
- **Local electronics stores:** Check clearance sections
- **University surplus:** Often has great deals
- **Recycling centers:** Find motors, sensors, power supplies

### **When to Buy**
- **Back-to-school season:** August-September (student discounts)
- **Black Friday:** November (electronics sales)
- **After holidays:** January (gift returns sell-off)
- **End of fiscal quarter:** Companies clearing old stock

### **Cost-Saving Strategies**

#### **Refrigerator Robotics**
- Salvage motors from old appliances
- Use recycled electronics components
- Repurpose old computer parts
- Check thrift stores for materials

#### **3D Printing**
- Print custom parts instead of buying
- Use local makerspaces (free/cheap printing)
- Share designs with learning community
- Modify existing open-source designs

#### **Educational Discounts**
- Student discounts on software/hardware
- Educational pricing on development boards
- University hardware lending programs
- Open-source software alternatives

---

## ✅ Pre-Purchase Checklist

### **Before Buying Computer**
- [ ] Check if current computer meets requirements
- [ ] Verify ROS 2 compatibility with your OS
- [ ] Test internet speed and reliability
- [ ] Review warranty and return policies

### **Before Buying Robot Components**
- [ ] Research component compatibility
- [ ] Check for required accessories (cables, adapters)
- [ ] Calculate total power requirements
- [ ] Plan for future expansion

### **Before Starting**
- [ ] Set up dedicated workspace
- [ ] Install necessary software
- [ ] Test all components individually
- [ ] Read safety guidelines for all equipment

---

## 🚀 Getting Started Guide

### **Day 1: Setup (30 minutes)**
1. **Assess current hardware** - Run compatibility checks
2. **Install ROS 2** - Follow installation guide
3. **Test basic setup** - Run hello-world example
4. **Bookmark resources** - Save important links

### **Week 1: Foundations**
1. **Complete Quarter 1 setup** - Ensure everything works
2. **Practice basic commands** - Get comfortable with CLI
3. **Try simple examples** - Build confidence
4. **Join community** - Get help when needed

### **Month 1: Comfort Level**
1. **Complete Quarter 1 content** - Solid foundation
2. **Start Quarter 2 preparation** - Check simulation setup
3. **Plan next hardware purchases** - Based on interest
4. **Set learning goals** - Define your robotics journey

---

## 📞 Getting Help

### **Hardware Issues**
- **Community Forums:** ROS Discourse, Reddit r/robotics
- **Manufacturer Support:** Component documentation and support
- **Local Groups:** Makerspaces, robotics clubs
- **Online Tutorials:** YouTube, Instructables, Hackaday

### **Setup Problems**
- **FAQ:** Common questions and solutions
- **Video Guides:** Step-by-step setup tutorials
- **One-on-One Help:** Community mentorship programs

---

## 🎉 Success Timeline

### **Immediate (Day 1)**
- ✅ Assess hardware compatibility
- ✅ Install basic software
- ✅ Run first ROS 2 program

### **Short-term (Week 1)**
- ✅ Complete Quarter 1 exercises
- ✅ Set up development environment
- ✅ Build confidence with basic concepts

### **Medium-term (Month 1-3)**
- ✅ Complete Quarter 2 simulation work
- ✅ Add perception capabilities
- ✅ Start planning robot build

### **Long-term (3+ months)**
- ✅ Build physical robot
- ✅ Implement advanced applications
- ✅ Join robotics community projects

---

**Remember:** The best hardware setup is the one you have right now! Start with what you have, learn the concepts, and gradually upgrade as your skills and interests grow. The most important investment is your time and dedication to learning! 🚀