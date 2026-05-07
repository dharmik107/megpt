import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Sparkles, MessageSquare, Mail, Trash2, Copy, Check, Menu, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
// import remarkGfm from 'remark-gfm';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth > 768);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { id: Date.now(), role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          history: messages.map(msg => ({ role: msg.role, content: msg.content }))
        }),
      });

      if (!response.ok) throw new Error('Failed to fetch response');

      const data = await response.json();
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', content: data.response }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', content: '### ⚠️ Error\nSorry, I encountered an error. Please make sure the backend is running.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const clearChat = () => {
    if (window.confirm('Are you sure you want to clear the conversation?')) {
      setMessages([]);
    }
  };

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="profile-img">
            <Bot size={40} color="#58a6ff" />
          </div>
          <h3>Dharmik Pansuriya</h3>
          <p>AI Engineer & Full-stack Developer</p>
        </div>

        <div className="sidebar-content">


          <div className="sidebar-section">
            <h4>Find Dharmik</h4>
            <div className="social-links">
              <a href="https://github.com/dharmik107" target="_blank" rel="noreferrer">
                <MessageSquare size={18} /> GitHub
              </a>
              <a href="https://linkedin.com/in/dharmik-pansuriya" target="_blank" rel="noreferrer">
                <MessageSquare size={18} /> LinkedIn
              </a>
              <a href="mailto:dharmikpansuriya107@gmail.com">
                <Mail size={18} /> Email
              </a>
            </div>
          </div>
        </div>

        <footer className="sidebar-footer">
          <button className="clear-button" onClick={clearChat}>
            <Trash2 size={16} /> Clear Conversation
          </button>
        </footer>
      </aside>

      {/* Main Chat Container */}
      <div className="chat-container">
        <header className="header">
          <div className="header-left">
            <button className="menu-toggle" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
              {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <h1>
              <Sparkles size={22} className="sparkle-icon" />
              MeGPT <span>Assistant</span>
            </h1>
          </div>
          <div className="status-indicator">
            <span className="pulse"></span>
            Online
          </div>
        </header>

        <main className="chat-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="bot-avatar-large">
                <Bot size={48} />
              </div>
              <h2>How can I help you today?</h2>
              <p>Ask me about Dharmik's projects, technical skills, or professional experience.</p>
              <div className="suggestion-grid">
                {[
                  "Tell me about Dharmik's projects",
                  "What are his top skills?",
                  "Where is he located?",
                  "Is he looking for internships?"
                ].map((text, i) => (
                  <button key={i} className="suggestion-chip" onClick={() => setInput(text)}>
                    {text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`message-wrapper ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                </div>
                <div className={`message-bubble ${msg.role}`}>
                  <div className="message-header">
                    <span>{msg.role === 'user' ? 'You' : 'MeGPT'}</span>
                    <button 
                      className="copy-button" 
                      onClick={() => copyToClipboard(msg.content, msg.id)}
                    >
                      {copiedId === msg.id ? <Check size={14} color="#238636" /> : <Copy size={14} />}
                    </button>
                  </div>
                  <div className="message-content">
                    <ReactMarkdown>
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            ))
          )}
          
          {isLoading && (
            <div className="message-wrapper assistant">
              <div className="message-avatar">
                <Bot size={20} />
              </div>
              <div className="typing-indicator-bubble">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </main>

        <footer className="input-area">
          <form onSubmit={handleSend} className="input-form">
            <div className="input-wrapper">
              <input
                type="text"
                placeholder="Message MeGPT..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isLoading}
              />
              <button type="submit" className="send-btn" disabled={!input.trim() || isLoading}>
                <Send size={18} />
              </button>
            </div>
          </form>
          <div className="footer-tagline">
            Powered by Groq, LangChain & Dharmik's Data
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
