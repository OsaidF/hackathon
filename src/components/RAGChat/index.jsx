import React, { useState, useRef, useEffect } from 'react';
import { X, Send, BookOpen, Loader2, ExternalLink, Sparkles } from 'lucide-react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import styles from './styles.module.css';
import ReactMarkdown from 'react-markdown';
import { API_URL } from '../../config';

export default function AskDocsButton() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [selectedText, setSelectedText] = useState('');
  const [showSelectionButton, setShowSelectionButton] = useState({ show: false, x: 0, y: 0 });
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Don't render on server side
  if (!ExecutionEnvironment.canUseDOM) {
    return null;
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    // Only run on client-side
    if (!ExecutionEnvironment.canUseDOM) return;
    
    // Load session ID from localStorage
    const storedSessionId = localStorage.getItem('docsSessionId');
    if (storedSessionId && storedSessionId !== 'undefined' && storedSessionId !== 'null') {
      setSessionId(storedSessionId);
      loadHistory(storedSessionId);
    }
  }, []);

  // Detect text selection
  useEffect(() => {
    if (!ExecutionEnvironment.canUseDOM) return;

    const handleSelection = () => {
      const selection = window.getSelection();
      const text = selection.toString().trim();
      
      if (text && text.length > 0 && text.length < 500) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        
        setSelectedText(text);
        setShowSelectionButton({
          show: true,
          x: rect.left + (rect.width / 2),
          y: rect.top + window.scrollY - 50
        });
      } else {
        setShowSelectionButton({ show: false, x: 0, y: 0 });
      }
    };

    document.addEventListener('mouseup', handleSelection);

    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  const loadHistory = async (sid) => {
    try {
      const response = await fetch(`${API_URL}/history/${sid}`);
      const data = await response.json();
      const formattedHistory = data.history.map(msg => ({
        role: msg.role,
        content: msg.content,
        sources: []
      }));
      setMessages(formattedHistory);
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          ...(sessionId && { session_id: sessionId })
        })
      });

      const data = await response.json();

      if (!sessionId && ExecutionEnvironment.canUseDOM) {
        setSessionId(data.session_id);
        localStorage.setItem('docsSessionId', data.session_id);
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        sources: data.sources || []
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        sources: []
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
    if (ExecutionEnvironment.canUseDOM) {
      localStorage.removeItem('docsSessionId');
    }
  };

  const askAboutSelection = () => {
    setIsOpen(true);
    setInput(`Explain this from the documentation: "${selectedText}"`);
    setShowSelectionButton({ show: false, x: 0, y: 0 });
    setTimeout(() => {
      inputRef.current?.focus();
    }, 100);
  };

  return (
    <>
      {/* Ask Docs Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={styles.askButton}
      >
        <BookOpen size={20} />
        Ask Docs
      </button>

      {/* Floating Selection Button */}
      {showSelectionButton.show && !isOpen && (
        <button
          onClick={askAboutSelection}
          className={styles.selectionButton}
          style={{
            position: 'absolute',
            top: `${showSelectionButton.y}px`,
            left: `${showSelectionButton.x}px`,
            transform: 'translateX(-50%)',
          }}
        >
          <Sparkles size={16} />
          Ask about selection
        </button>
      )}

      {/* Modal Overlay */}
      {isOpen && (
        <div className={styles.modalOverlay}>
          {/* Modal Container */}
          <div className={styles.modalContainer}>
            {/* Header */}
            <div className={styles.header}>
              <div className={styles.headerContent}>
                <div className={styles.iconWrapper}>
                  <BookOpen size={24} />
                </div>
                <div className={styles.headerText}>
                  <h2>Ask Documentation</h2>
                  <p>Get instant answers from our docs</p>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className={styles.closeButton}
              >
                <X size={20} />
              </button>
            </div>

            {/* Messages Area */}
            <div className={styles.messagesArea}>
              {messages.length === 0 ? (
                <div className={styles.emptyState}>
                  <div className={styles.emptyIcon}>
                    <BookOpen size={48} />
                  </div>
                  <h3>How can I help you?</h3>
                  <p>
                    Ask me anything about the documentation. I'll search through our docs to find the most relevant information.
                  </p>
                  <div className={styles.suggestions}>
                    {[
                      "How do I get started?",
                      "What are the API endpoints?",
                      "How do I authenticate?"
                    ].map((suggestion, idx) => (
                      <button
                        key={idx}
                        onClick={() => {
                          setInput(suggestion);
                          inputRef.current?.focus();
                        }}
                        className={styles.suggestionButton}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                  <div className={styles.protipcontainer}>
                    <div className={styles.protipheader}>
                      <Sparkles size={16} />
                      Pro tip
                    </div>
                    <p className={styles.protiptext}>
                      Select any text on the page and click "Ask about selection" to learn more!
                    </p>
                  </div>
                </div>
              ) : (
                <>
                  <div className={styles.messagesList}>
                    {messages.map((message, idx) => (
                      <div
                        key={idx}
                        className={`${styles.message} ${styles[message.role]}`}
                      >
                        <div className={`${styles.avatar} ${styles[message.role]}`}>
                          {message.role === 'user' ? (
                            'U'
                          ) : (
                            <BookOpen size={16} />
                          )}
                        </div>
                        <div className={styles.messageContent}>
                          <div className={`${styles.messageBubble} ${styles[message.role]}`}>
                            {message.role === 'assistant' ? (
                              <div className={styles.markdown}>
                                <ReactMarkdown
                                  components={{
                                    code({node, inline, className, children, ...props}) {
                                      return inline ? (
                                        <code className={styles.inlineCode} {...props}>
                                          {children}
                                        </code>
                                      ) : (
                                        <pre className={styles.codeBlock}>
                                          <code className={className} {...props}>
                                            {children}
                                          </code>
                                        </pre>
                                      );
                                    },
                                    a({node, children, ...props}) {
                                      return (
                                        <a {...props} target="_blank" rel="noopener noreferrer" className={styles.markdownLink}>
                                          {children}
                                        </a>
                                      );
                                    },
                                    p({children}) {
                                      return <p className={styles.paragraph}>{children}</p>;
                                    },
                                    ul({children}) {
                                      return <ul className={styles.list}>{children}</ul>;
                                    },
                                    ol({children}) {
                                      return <ol className={styles.orderedList}>{children}</ol>;
                                    },
                                    li({children}) {
                                      return <li className={styles.listItem}>{children}</li>;
                                    },
                                    h1({children}) {
                                      return <h1 className={styles.heading1}>{children}</h1>;
                                    },
                                    h2({children}) {
                                      return <h2 className={styles.heading2}>{children}</h2>;
                                    },
                                    h3({children}) {
                                      return <h3 className={styles.heading3}>{children}</h3>;
                                    },
                                    blockquote({children}) {
                                      return <blockquote className={styles.blockquote}>{children}</blockquote>;
                                    },
                                  }}
                                >
                                  {message.content}
                                </ReactMarkdown>
                              </div>
                            ) : (
                              message.content
                            )}
                          </div>

                          {/* Sources */}
                          {message.sources && message.sources.length > 0 && (
                            <div className={styles.sources}>
                              <div className={styles.sourcesLabel}>Sources:</div>
                              <div className={styles.sourcesList}>
                                {message.sources.map((source, sidx) => (
                                  <a
                                    key={sidx}
                                    href={source.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className={styles.sourceLink}
                                  >
                                    <ExternalLink size={12} />
                                    {source.title}
                                  </a>
                                ))}
                              </div>
                            </div>
                          )}  
                        </div>
                      </div>
                    ))}

                    {isLoading && (
                      <div className={styles.loadingMessage}>
                        <div className={`${styles.avatar} ${styles.assistant}`}>
                          <BookOpen size={16} />
                        </div>
                        <div className={styles.loadingBubble}>
                          <Loader2 size={16} />
                          Searching docs...
                        </div>
                      </div>
                    )}
                  </div>
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Input Area */}
            <div className={styles.inputArea}>
              <div className={styles.inputWrapper}>
                <div className={styles.inputContainer}>
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask a question..."
                    rows={1}
                    className={styles.textarea}
                    style={{ minHeight: '48px' }}
                  />
                </div>
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || isLoading}
                  className={styles.sendButton}
                >
                  <Send size={20} />
                </button>
              </div>
              {messages.length > 0 && (
                <button
                  onClick={clearChat}
                  className={styles.clearButton}
                >
                  Clear conversation
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}