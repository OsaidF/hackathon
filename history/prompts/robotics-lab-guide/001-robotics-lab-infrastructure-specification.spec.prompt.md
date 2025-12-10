---
id: 001
title: robotics lab infrastructure specification
stage: spec
date: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20250929
feature: robotics-lab-guide
branch: 001-robotics-lab-guide
user: user
command: /sp.specify
labels: ["infrastructure","hardware","robotics","education","budget-planning"]
links:
  spec: specs/001-robotics-lab-guide/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-robotics-lab-guide/spec.md
  - specs/001-robotics-lab-guide/checklists/requirements.md
tests:
  - Validation completed: All checklist items pass, specification ready for planning
---

## Prompt

Physical AI & Humanoid Robotics Quarter Course Infrastructure and Lab Setup Guide

Target audience: Educational institutions, bootcamp administrators, and technical program directors planning to launch embodied AI/robotics training programs

Focus: Hardware requirements, lab architecture, and cost-effective deployment strategies for teaching ROS 2, simulation platforms, and NVIDIA Isaac-based humanoid robotics

Success criteria:
- Reader can make informed CapEx vs OpEx decisions for lab setup
- Clear hardware specifications for each of four teaching modules
- Budget-conscious alternatives identified with tradeoffs explained
- Deployment-ready shopping lists for student workstations and edge kits
- Reader understands why specific GPU/compute requirements exist

Constraints:
- Format: Technical specification document with itemized hardware lists
- Budget tiers: Economy (~$700/student), Mid-range (~$3,000), Premium (~$16,000+)
- Timeline considerations: 13-week quarter structure
- Platform requirements: Ubuntu 22.04 LTS, ROS 2 Humble/Iron, NVIDIA Isaac Sim

Four Module Breakdown:

MODULE 1: The Robotic Nervous System (ROS 2)
Hardware needs:
- Development workstation with Linux (Ubuntu 22.04 LTS)
- Minimum: 16GB RAM, quad-core CPU
- Focus: Software middleware, no GPU required for basic ROS 2 nodes
- Edge deployment: Jetson Orin Nano for rclpy controller testing

MODULE 2: The Digital Twin (Gazebo & Unity)
Hardware needs:
- Physics simulation workstation
- CPU: Intel i7 13th gen+ or AMD Ryzen 9 (physics calculations)
- GPU: RTX 3060 minimum for Gazebo rendering
- RAM: 32GB minimum (64GB recommended for Unity high-fidelity scenes)
- Storage: 500GB+ SSD for USD assets and simulation environments

MODULE 3: The AI-Robot Brain (NVIDIA Isaac)
Hardware needs:
- High-performance simulation rig
- GPU: RTX 4070 Ti (12GB VRAM) minimum, RTX 4090 (24GB) ideal
- Required: RTX architecture for ray tracing in Isaac Sim
- RAM: 64GB DDR5 (mandatory for photorealistic simulation)
- Edge hardware: Jetson Orin Nano/NX for Isaac ROS deployment
- Sensors: Intel RealSense D435i for VSLAM/perception testing

MODULE 4: Vision-Language-Action (VLA)
Hardware needs:
- Combined inference system (simulation + LLM)
- GPU: Same RTX workstation from Module 3 (shared VRAM load)
- Edge deployment: Jetson Orin with RealSense camera
- Audio: USB microphone array (ReSpeaker) for Whisper voice commands
- Network: Low-latency connection for voice-to-action pipeline

Physical Robot Options:
- Budget proxy: Unitree Go2 quadruped (~$1,800-$3,000)
- Miniature humanoid: Unitree G1 (~$16,000) or Hiwonder TonyPi (~$600)
- Premium: Unitree G1 for full sim-to-real humanoid deployment

Not building:
- Comprehensive robotics curriculum (focuses only on infrastructure)
- Software installation guides or ROS 2 tutorials
- Detailed lesson plans or assessment rubrics
- Cloud cost optimization strategies (mentions AWS/Azure options but doesn't deep-dive)
- Comparison of alternative simulation platforms beyond Gazebo/Unity/Isaac
- Safety protocols or lab management procedures
- Student prerequisite knowledge requirements

## Response snapshot

Successfully created comprehensive specification for Physical AI & Humanoid Robotics Quarter Course Infrastructure and Lab Setup Guide. Created feature branch 001-robotics-lab-guide with detailed user stories covering infrastructure planning, hardware configuration, budget optimization, and edge deployment. Generated 12 functional requirements and 7 measurable success criteria. Specification passes all validation criteria with no [NEEDS CLARIFICATION] markers and is ready for planning phase.

## Outcome

- ✅ Impact: Educational institutions can now make informed hardware procurement decisions for robotics programs with clear budget tier guidance and technical specifications
- 🧪 Tests: Validation completed - all checklist items pass, specification ready for planning
- 📁 Files: Created specs/001-robotics-lab-guide/spec.md and specs/001-robotics-lab-guide/checklists/requirements.md
- 🔁 Next prompts: /sp.clarify or /sp.plan to proceed with implementation planning
- 🧠 Reflection: Successfully translated complex technical requirements into user-focused specification with measurable outcomes

## Evaluation notes (flywheel)

- Failure modes observed: None - all validation criteria passed
- Graders run and results (PASS/FAIL): PASS - Specification meets all quality standards
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A - specification is complete and ready for planning phase