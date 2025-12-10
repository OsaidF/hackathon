import React from 'react';
import styles from './styles.module.css';

const ComplexityIndicator = ({ level, title, description, prerequisites = [] }) => {
  const getLevelConfig = (level) => {
    const configs = {
      beginner: {
        color: '#4CAF50', // Green
        icon: '🔵',
        label: 'Beginner',
        description: 'Basic concepts, no prerequisites required'
      },
      intermediate: {
        color: '#2196F3', // Blue
        icon: '🟢',
        label: 'Intermediate',
        description: 'Some background knowledge helpful'
      },
      advanced: {
        color: '#FF9800', // Orange
        icon: '🟡',
        label: 'Advanced',
        description: 'Requires solid foundation in the area'
      },
      expert: {
        color: '#F44336', // Red
        icon: '🔴',
        label: 'Expert',
        description: 'Cutting-edge research topics'
      }
    };
    return configs[level] || configs.beginner;
  };

  const config = getLevelConfig(level);

  return (
    <div className={styles.complexityIndicator} style={{ borderColor: config.color }}>
      <div className={styles.header}>
        <span className={styles.icon}>{config.icon}</span>
        <span className={styles.label} style={{ color: config.color }}>
          {config.label}
        </span>
      </div>
      {title && (
        <div className={styles.title}>
          <strong>{title}</strong>
        </div>
      )}
      {description && (
        <div className={styles.description}>
          {description}
        </div>
      )}
      {prerequisites && prerequisites.length > 0 && (
        <div className={styles.prerequisites}>
          <strong>Prerequisites:</strong>
          <ul>
            {prerequisites.map((prereq, index) => (
              <li key={index}>{prereq}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ComplexityIndicator;