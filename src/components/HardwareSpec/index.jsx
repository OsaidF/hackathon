import React, { useState } from 'react';
import useIsBrowser from '@docusaurus/useIsBrowser';
import './styles.module.css';

export default function HardwareSpec({
  title,
  description,
  costRange,
  difficulty,
  timeToStart,
  specs = [],
  included = [],
  notIncluded = [],
  optional = [],
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isBrowser = useIsBrowser();

  const getDifficultyColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'beginner': return 'var(--ifm-color-success)';
      case 'intermediate': return 'var(--ifm-color-warning)';
      case 'advanced': return 'var(--ifm-color-danger)';
      default: return 'var(--ifm-color-info)';
    }
  };

  const getDifficultyIcon = (level) => {
    switch (level?.toLowerCase()) {
      case 'beginner': return '🟢';
      case 'intermediate': return '🟡';
      case 'advanced': return '🔴';
      default: return '🔵';
    }
  };

  return (
    <div className="hardware-spec-container">
      <div className="hardware-spec-header">
        <div className="hardware-spec-title-section">
          <h3 className="hardware-spec-title">{title}</h3>
          <p className="hardware-spec-description">{description}</p>
        </div>

        <div className="hardware-spec-badges">
          <div className="hardware-spec-badge cost-badge">
            <span className="badge-icon">💰</span>
            <span className="badge-text">{costRange}</span>
          </div>

          <div className="hardware-spec-badge difficulty-badge" style={{ borderColor: getDifficultyColor(difficulty) }}>
            <span className="badge-icon">{getDifficultyIcon(difficulty)}</span>
            <span className="badge-text">{difficulty}</span>
          </div>

          <div className="hardware-spec-badge time-badge">
            <span className="badge-icon">⏱️</span>
            <span className="badge-text">{timeToStart}</span>
          </div>
        </div>
      </div>

      {(specs.length > 0 || included.length > 0 || notIncluded.length > 0 || optional.length > 0) && (
        <div className="hardware-spec-content">
          <button
            className="hardware-spec-toggle"
            onClick={() => setIsExpanded(!isExpanded)}
            aria-expanded={isExpanded}
          >
            <span>{isExpanded ? 'Hide Details' : 'Show Details'}</span>
            <span className={`toggle-icon ${isExpanded ? 'expanded' : ''}`}>▼</span>
          </button>

          {isExpanded && (
            <div className="hardware-spec-details">
              {specs.length > 0 && (
                <div className="spec-section">
                  <h4>📋 Technical Specifications</h4>
                  <div className="spec-grid">
                    {specs.map((spec, index) => (
                      <div key={index} className="spec-item">
                        <span className="spec-label">{spec.label}:</span>
                        <span className="spec-value">{spec.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {included.length > 0 && (
                <div className="spec-section">
                  <h4>✅ What's Included</h4>
                  <ul className="spec-list included-list">
                    {included.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {notIncluded.length > 0 && (
                <div className="spec-section">
                  <h4>❌ What's Not Included</h4>
                  <ul className="spec-list not-included-list">
                    {notIncluded.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {optional.length > 0 && (
                <div className="spec-section">
                  <h4>⚙️ Optional Add-ons</h4>
                  <ul className="spec-list optional-list">
                    {optional.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}