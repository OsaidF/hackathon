import React, { useState } from 'react';
import styles from './styles.module.css';

const DeepDive = ({ title, children, defaultExpanded = false }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className={styles.deepDive}>
      <button
        className={styles.header}
        onClick={toggleExpanded}
        aria-expanded={isExpanded}
      >
        <span className={styles.icon}>🔬</span>
        <span className={styles.title}>Deep Dive: {title}</span>
        <span className={`${styles.arrow} ${isExpanded ? styles.expanded : ''}`}>
          ▼
        </span>
      </button>
      <div className={`${styles.content} ${isExpanded ? styles.expanded : ''}`}>
        {children}
      </div>
    </div>
  );
};

export default DeepDive;