# Physical AI & Humanoid Robotics - Educational Book

A comprehensive educational resource on Physical AI and Humanoid Robotics, built with Docusaurus and deployed to GitHub Pages.

## Overview

This book provides a structured learning progression across four quarters, from ROS 2 foundations to advanced embodied AI applications. It's designed for students, educators, researchers, and professionals interested in the intersection of artificial intelligence and humanoid robotics.

## Book Structure

### Quarter 1: The Robotic Nervous System - Foundations and Middleware
- ROS 2 architecture and distributed systems
- Communication patterns and fundamental robotics concepts
- Hardware introduction and basic setup

### Quarter 2: The Digital Twin - Simulation and Virtual Environments
- Physics simulation and digital twins
- Gazebo and Unity simulation platforms
- Simulation-to-real transfer concepts

### Quarter 3: The AI-Robot Brain - Perception and Intelligence
- Computer vision and sensor fusion
- Perception algorithms and AI integration
- NVIDIA Isaac Sim and edge deployment

### Quarter 4: Vision-Language-Action - Advanced Embodied AI
- Multimodal AI and human-robot interaction
- Vision-language models and voice control
- Advanced embodied intelligence applications

## Features

- **Progressive Learning**: Structured content building from foundations to advanced topics
- **Interactive Examples**: Code snippets with environment setup instructions
- **Hardware Context**: Practical guidance with system requirements
- **Up-to-Date Documentation**: Integrated Context7 MCP for latest robotics library documentation
- **Academic Rigor**: Peer-reviewed sources with proper citations

## Setup

### Prerequisites

- Node.js 18.0 or higher
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/humanoid-robotics.git
   cd humanoid-robotics
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Development

Start the development server:
```bash
npm start
```

Open [http://localhost:3000](http://localhost:3000) to view the site.

### Build

Build the static site for production:
```bash
npm run build
```

The built files will be in the `build` directory.

### Linting and Formatting

```bash
# Run ESLint
npm run lint

# Fix ESLint issues
npm run lint:fix

# Format code with Prettier
npm run format

# Check code formatting
npm run format:check
```

## Contributing

This project follows the Spec-Driven Development methodology. All changes should:

1. Start with specification updates in the `specs/` directory
2. Follow the established task breakdown in `specs/001-robotics-lab-guide/tasks.md`
3. Maintain academic rigor with proper citations
4. Ensure accessibility compliance (WCAG 2.1 AA)

## Architecture

- **Platform**: Docusaurus 3.9.2 with React 18
- **Content**: MDX for rich documentation with interactive components
- **Deployment**: GitHub Pages for static hosting
- **Documentation**: Context7 MCP integration for up-to-date library references

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this educational resource in your work, please cite:

```bibtex
@book{humanoid_robotics_2025,
  title={Physical AI & Humanoid Robotics: Comprehensive Educational Guide},
  author={[Author Names]},
  year={2025},
  publisher={GitHub},
  url={https://your-username.github.io/humanoid-robotics/}
}
```

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Review the specification in `specs/001-robotics-lab-guide/spec.md`
- Check the implementation tasks in `specs/001-robotics-lab-guide/tasks.md`