import React, { useState } from 'react';
import useIsBrowser from '@docusaurus/useIsBrowser';
import { usePrismTheme } from '@docusaurus/theme-common';
import { Playground as DocusaurusPlayground } from '@docusaurus/theme-common/internal';

// Context7 MCP integration placeholder
// This will be enhanced to fetch code examples from robotics libraries
const fetchContext7Code = async (library, example) => {
  // TODO: Implement Context7 MCP integration
  // const response = await fetch(`/api/context7/${library}/${example}`);
  // return response.json();
  return null;
};

export default function CodeBlock({
  children,
  language = 'python',
  title,
  showLineNumbers = true,
  highlightLines = [],
  executable = false,
  context7Library = null,
  context7Example = null,
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [context7Code, setContext7Code] = useState(null);
  const [loadingContext7, setLoadingContext7] = useState(false);
  const isBrowser = useIsBrowser();
  const prismTheme = usePrismTheme();

  const loadContext7Code = async () => {
    if (context7Library && context7Example && !context7Code) {
      setLoadingContext7(true);
      try {
        const code = await fetchContext7Code(context7Library, context7Example);
        setContext7Code(code);
      } catch (error) {
        console.error('Failed to load Context7 code:', error);
      } finally {
        setLoadingContext7(false);
      }
    }
  };

  const handleContext7Load = () => {
    setIsExpanded(true);
    loadContext7Code();
  };

  // Robust function to extract code string from various children formats
  const getCodeString = (children) => {
    if (typeof children === 'string') {
      return children;
    }

    if (Array.isArray(children)) {
      return children.map(child => getCodeString(child)).join('');
    }

    if (children && typeof children === 'object') {
      if (children.props && children.props.children) {
        return getCodeString(children.props.children);
      }
      return children.toString();
    }

    return String(children || '');
  };

  const codeString = getCodeString(children);

  return (
    <div className={`code-block-container ${language}`}>
      {title && (
        <div className="code-block-header">
          <span className="code-block-title">{title}</span>
          {executable && (
            <span className="code-block-executable-badge">▶ Runnable</span>
          )}
        </div>
      )}

      <div className="code-block-content">
        <pre className={`language-${language}`}>
          <code
            className={`language-${language}`}
            style={{
              ...(highlightLines.length > 0 && {
                '.highlight': {
                  background: prismTheme.plain.backgroundColor,
                  borderLeft: '3px solid var(--ifm-color-primary)',
                  paddingLeft: '0.5rem',
                },
              }),
            }}
          >
            {codeString.split('\n').map((line, index) => (
              <div
                key={index}
                className={`code-line ${
                  highlightLines.includes(index + 1) ? 'highlight' : ''
                }`}
              >
                {showLineNumbers && (
                  <span className="line-number">{index + 1}</span>
                )}
                <span className="line-content">{line}</span>
              </div>
            ))}
          </code>
        </pre>
      </div>

      {context7Library && context7Example && (
        <div className="context7-code-section">
          {!isExpanded ? (
            <button
              className="context7-load-button"
              onClick={handleContext7Load}
              disabled={loadingContext7}
            >
              📚 Load Latest {context7Library} Example
            </button>
          ) : (
            <div className="context7-code-content">
              <div className="context7-header">
                <span>📚 Latest from {context7Library}</span>
                {loadingContext7 && <span className="loading">Loading...</span>}
              </div>
              {context7Code && (
                <div className="context7-code">
                  <pre>
                    <code>{context7Code}</code>
                  </pre>
                  <div className="context7-source">
                    Source: Context7 MCP - {context7Library}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {executable && (
        <div className="code-block-execution">
          <div className="execution-note">
            💡 This code is executable. Copy and run it in your ROS 2 environment.
          </div>
          <button
            className="copy-button"
            onClick={() => {
              navigator.clipboard.writeText(codeString);
            }}
          >
            📋 Copy Code
          </button>
        </div>
      )}
    </div>
  );
}

// Specialized components for robotics code examples
export function PythonCode(props) {
  return <CodeBlock {...props} language="python" />;
}

export function BashCode(props) {
  return <CodeBlock {...props} language="bash" showLineNumbers={false} />;
}

export function ROS2Code(props) {
  return (
    <CodeBlock
      {...props}
      language="python"
      title="ROS 2 Example"
      executable={true}
    />
  );
}

export function SimulationCode(props) {
  return (
    <CodeBlock
      {...props}
      language="python"
      title="Gazebo/Unity Simulation"
      context7Library="gazebo"
    />
  );
}

export function AICode(props) {
  return (
    <CodeBlock
      {...props}
      language="python"
      title="AI/ML Example"
      context7Library="pytorch"
    />
  );
}