import React from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';

import styles from './index.module.css';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <main className={styles.homepage}>
      {/* Hero Section */}
      <section className={styles.hero}>
        <div className={styles.heroBackground}>
          <div className={styles.heroGrid}></div>
          <div className={styles.heroParticles}></div>
        </div>

        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            Physical AI &
            <span className={styles.heroTitleAccent}> Humanoid Robotics</span>
          </h1>

          <p className={styles.heroSubtitle}>
            Comprehensive Educational Guide from ROS 2 Foundations to Advanced Embodied AI
          </p>

          <p className={styles.heroDescription}>
            Master the intersection of artificial intelligence and physical robotics
            through hands-on learning with real-world applications.
          </p>

          <div className={styles.heroActions}>
            <Link
              to="/docs/intro"
              className={clsx('button button--primary button--lg', styles.primaryButton)}
            >
              Start Learning Journey
            </Link>

            <Link
              to="/docs/hardware/minimum-requirements"
              className={clsx('button button--secondary button--lg', styles.secondaryButton)}
            >
              View Requirements
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className={styles.features}>
        <div className="container">
          <div className={styles.featuresHeader}>
            <h2 className={styles.featuresTitle}>
              Why Choose This Guide?
            </h2>
            <p className={styles.featuresSubtitle}>
              A comprehensive learning experience designed for diverse backgrounds
            </p>
          </div>

          <div className={styles.featuresGrid}>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <svg className={styles.featureSvg} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                </svg>
              </div>
              <h3 className={styles.featureTitle}>Progressive Learning</h3>
              <p className={styles.featureDescription}>
                Four-quarter structure builds from fundamentals to advanced embodied AI topics
              </p>
            </div>

            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <svg className={styles.featureSvg} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
                </svg>
              </div>
              <h3 className={styles.featureTitle}>Hands-On Focus</h3>
              <p className={styles.featureDescription}>
                Practical examples with real robotics platforms and simulation environments
              </p>
            </div>

            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <svg className={styles.featureSvg} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                  <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                </svg>
              </div>
              <h3 className={styles.featureTitle}>Academic Rigor</h3>
              <p className={styles.featureDescription}>
                Peer-reviewed sources with proper citations and current state-of-the-art integration
              </p>
            </div>

            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <svg className={styles.featureSvg} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
                  <line x1="8" y1="21" x2="16" y2="21"/>
                  <line x1="12" y1="17" x2="12" y2="21"/>
                </svg>
              </div>
              <h3 className={styles.featureTitle}>Hardware Context</h3>
              <p className={styles.featureDescription}>
                Clear understanding of computational requirements and platform compatibility
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Path Section */}
      <section className={styles.learningPath}>
        <div className="container">
          <div className={styles.learningPathHeader}>
            <h2 className={styles.learningPathTitle}>Learning Journey</h2>
            <p className={styles.learningPathSubtitle}>
              Four quarters taking you from robotics basics to advanced embodied AI
            </p>
          </div>

          <div className={styles.quarterGrid}>
            <div className={styles.quarterCard}>
              <div className={styles.quarterNumber}>Q1</div>
              <h3 className={styles.quarterTitle}>The Robotic Nervous System</h3>
              <p className={styles.quarterDescription}>
                ROS 2 foundations and distributed robotics middleware
              </p>
              <ul className={styles.quarterTopics}>
                <li>ROS 2 Architecture</li>
                <li>Communication Patterns</li>
                <li>Hardware Interfaces</li>
                <li>Basic Programming</li>
              </ul>
              <Link to="/docs/quarter-1/01-robotics-overview" className={styles.quarterLink}>
                Start Quarter 1 →
              </Link>
            </div>

            <div className={styles.quarterCard}>
              <div className={styles.quarterNumber}>Q2</div>
              <h3 className={styles.quarterTitle}>The Digital Twin</h3>
              <p className={styles.quarterDescription}>
                Simulation and virtual environments for safe testing
              </p>
              <ul className={styles.quarterTopics}>
                <li>Physics Simulation</li>
                <li>Gazebo & Unity</li>
                <li>Digital Twins</li>
                <li>Sim-to-Real Transfer</li>
              </ul>
              <Link to="/docs/quarter-2/06-physics-simulation" className={styles.quarterLink}>
                Explore Quarter 2 →
              </Link>
            </div>

            <div className={styles.quarterCard}>
              <div className={styles.quarterNumber}>Q3</div>
              <h3 className={styles.quarterTitle}>The AI-Robot Brain</h3>
              <p className={styles.quarterDescription}>
                Perception, computer vision, and AI model deployment
              </p>
              <ul className={styles.quarterTopics}>
                <li>Computer Vision</li>
                <li>Sensor Fusion</li>
                <li>Edge AI Deployment</li>
                <li>NVIDIA Isaac Sim</li>
              </ul>
              <Link to="/docs/quarter-3/11-computer-vision" className={styles.quarterLink}>
                Master Quarter 3 →
              </Link>
            </div>

            <div className={styles.quarterCard}>
              <div className={styles.quarterNumber}>Q4</div>
              <h3 className={styles.quarterTitle}>Advanced Embodied AI</h3>
              <p className={styles.quarterDescription}>
                Vision-language models and human-robot interaction
              </p>
              <ul className={styles.quarterTopics}>
                <li>Multimodal AI</li>
                <li>Vision-Language Models</li>
                <li>Human-Robot Interaction</li>
                <li>Voice Control</li>
              </ul>
              <Link to="/docs/quarter-4/16-multimodal-ai" className={styles.quarterLink}>
                Advance to Quarter 4 →
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Target Audience Section */}
      <section className={styles.audience}>
        <div className="container">
          <div className={styles.audienceHeader}>
            <h2 className={styles.audienceTitle}>Designed for Every Background</h2>
            <p className={styles.audienceSubtitle}>
              Tailored learning paths that leverage your existing expertise
            </p>
          </div>

          <div className={styles.audienceGrid}>
            <div className={styles.audienceCard}>
              <div className={styles.audienceIcon}>👨‍💻</div>
              <h3 className={styles.audienceRole}>Software Engineers</h3>
              <p className={styles.audienceDesc}>
                Leverage your programming skills to master real-time robotics systems
              </p>
            </div>

            <div className={styles.audienceCard}>
              <div className={styles.audienceIcon}>⚙️</div>
              <h3 className={styles.audienceRole}>Mechanical Engineers</h3>
              <p className={styles.audienceDesc}>
                Apply your physical systems knowledge to robotics software architecture
              </p>
            </div>

            <div className={styles.audienceCard}>
              <div className={styles.audienceIcon}>🤖</div>
              <h3 className={styles.audienceRole}>Computer Scientists</h3>
              <p className={styles.audienceDesc}>
                Bridge algorithmic expertise with real-world robotics constraints
              </p>
            </div>

            <div className={styles.audienceCard}>
              <div className={styles.audienceIcon}>📊</div>
              <h3 className={styles.audienceRole}>Data Scientists</h3>
              <p className={styles.audienceDesc}>
                Apply ML expertise to real-time sensor data and adaptive systems
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className={styles.cta}>
        <div className="container">
          <div className={styles.ctaContent}>
            <h2 className={styles.ctaTitle}>
              Ready to Shape the Future of Robotics?
            </h2>
            <p className={styles.ctaDescription}>
              Join thousands of learners mastering the intersection of AI and physical robotics
            </p>
            <div className={styles.ctaActions}>
              <Link
                to="/docs/intro"
                className={clsx('button button--primary button--lg', styles.ctaButton)}
              >
                Start Your Journey
              </Link>
              <Link
                to="/docs/hardware/minimum-requirements"
                className={styles.ctaLink}
              >
                Check Hardware Requirements →
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}