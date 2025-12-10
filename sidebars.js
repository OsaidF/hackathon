/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const tutorialSidebar = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Quarter 1: ROS 2 Foundations',
      collapsible: true,
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'quarter-1/index',
          label: 'Quarter 1 Overview',
        },
        {
          type: 'doc',
          id: 'quarter-1/robotics-overview',
          label: '1. Robotics Overview',
        },
        {
          type: 'doc',
          id: 'quarter-1/ros2-architecture',
          label: '2. ROS 2 Architecture',
        },
        {
          type: 'doc',
          id: 'quarter-1/communication-patterns',
          label: '3. Communication Patterns',
        },
        {
          type: 'doc',
          id: 'quarter-1/distributed-systems',
          label: '4. Distributed Systems',
        },
        {
          type: 'doc',
          id: 'quarter-1/hardware-intro',
          label: '5. Hardware Introduction',
        },
      ],
    },
    {
      type: 'category',
      label: 'Quarter 2: Simulation and Digital Worlds',
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'quarter-2/index',
          label: 'Quarter 2 Overview',
        },
        {
          type: 'doc',
          id: 'quarter-2/physics-simulation',
          label: '6. Physics Simulation',
        },
        {
          type: 'doc',
          id: 'quarter-2/gazebo-fundamentals',
          label: '7. Gazebo Fundamentals',
        },
        {
          type: 'doc',
          id: 'quarter-2/unity-robotics',
          label: '8. Unity Robotics',
        },
        {
          type: 'doc',
          id: 'quarter-2/digital-twins',
          label: '9. Digital Twins',
        },
        {
          type: 'doc',
          id: 'quarter-2/sim2real',
          label: '10. Sim2Real',
        },
      ],
    },
    {
      type: 'category',
      label: 'Quarter 3: Perception and Intelligence',
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'quarter-3/index',
          label: 'Quarter 3 Overview',
        },
        {
          type: 'doc',
          id: 'quarter-3/computer-vision',
          label: '11. Computer Vision',
        },
        {
          type: 'doc',
          id: 'quarter-3/sensor-fusion',
          label: '12. Sensor Fusion',
        },
        {
          type: 'doc',
          id: 'quarter-3/perception-algorithms',
          label: '13. Perception Algorithms',
        },
        {
          type: 'doc',
          id: 'quarter-3/isaac-sim',
          label: '14. Isaac Sim',
        },
        {
          type: 'doc',
          id: 'quarter-3/edge-deployment',
          label: '15. Edge Deployment',
        },
        {
          type: 'doc',
          id: 'quarter-3/context7-mcp-integration',
          label: 'Context7 Integration (OpenCV & Isaac Sim)',
        },
      ],
    },
    {
      type: 'category',
      label: 'Quarter 4: Multimodal AI and Human-Robot Interaction',
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'quarter-4/index',
          label: 'Quarter 4 Overview',
        },
        {
          type: 'doc',
          id: 'quarter-4/multimodal-ai',
          label: '16. Multimodal AI',
        },
        {
          type: 'doc',
          id: 'quarter-4/vision-language',
          label: '17. Vision-Language Models',
        },
        {
          type: 'doc',
          id: 'quarter-4/human-robot-interaction',
          label: '18. Human-Robot Interaction',
        },
        {
          type: 'doc',
          id: 'quarter-4/voice-control',
          label: '19. Voice Control',
        },
        {
          type: 'doc',
          id: 'quarter-4/future-directions',
          label: '20. Future Directions',
        },
        {
          type: 'doc',
          id: 'quarter-4/context7-ml-integration',
          label: 'Context7 Integration (AI/ML Frameworks)',
        },
      ],
    },
    {
      type: 'category',
      label: 'Hardware Requirements and Setup',
      collapsible: true,
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'hardware/index',
          label: 'Hardware Setup Overview',
        },
        {
          type: 'doc',
          id: 'hardware/minimum-requirements',
          label: 'Minimum Requirements',
        },
        {
          type: 'doc',
          id: 'hardware/recommended-setups',
          label: 'Recommended Setups',
        },
        {
          type: 'doc',
          id: 'hardware/platform-compatibility',
          label: 'Platform Compatibility',
        },
        {
          type: 'doc',
          id: 'hardware/installation-guides',
          label: 'Installation Guides',
        },
        {
          type: 'doc',
          id: 'hardware/budget-analysis',
          label: 'Budget Analysis',
        },
        {
          type: 'doc',
          id: 'hardware/cloud-alternatives',
          label: 'Cloud Alternatives',
        },
      ],
    },
    {
      type: 'category',
      label: 'Resources',
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'resources/references',
          label: 'References',
        },
        {
          type: 'doc',
          id: 'resources/glossary',
          label: 'Glossary',
        },
        {
          type: 'doc',
          id: 'resources/code-examples/README',
          label: 'Code Examples',
        },
        {
          type: 'doc',
          id: 'resources/state-of-the-art-surveys',
          label: 'State-of-the-Art Surveys',
        },
        {
          type: 'doc',
          id: 'resources/experimental-setup-guides',
          label: 'Experimental Setup Guides',
        },
        {
          type: 'doc',
          id: 'resources/bibliography',
          label: 'Bibliography (APA 7th)',
        },
        {
          type: 'doc',
          id: 'resources/research-methodology',
          label: 'Research Methodology',
        },
        {
          type: 'doc',
          id: 'resources/citation-system',
          label: 'Citation System',
        },
        {
          type: 'doc',
          id: 'resources/prerequisite-guides',
          label: 'Prerequisite Guides',
        },
        {
          type: 'doc',
          id: 'resources/transition-guides',
          label: 'Transition Guides',
        },
      ],
    },
  ],
};

export default tutorialSidebar;