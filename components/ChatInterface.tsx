import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, RagContext } from '../types';
import { generateRAGResponse } from '../services/geminiService';
import { Send, Bot, User, Database, ChevronDown, ChevronRight, Sparkles } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'system',
      content: 'Hello! I am your AI Financial Analyst. I have access to the latest stock data for AAPL, NVDA, GOOGL, AMZN, and NFLX. Ask me about trends, comparisons, or specific price action.',
      timestamp: Date.now()
    }
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isProcessing) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsProcessing(true);

    // Add a placeholder "Thinking" message
    const thinkingId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, {
      id: thinkingId,
      role: 'model',
      content: 'Analyzing market data...',
      isThinking: true,
      timestamp: Date.now()
    }]);

    try {
      const response = await generateRAGResponse(userMsg.content, messages);
      
      // Replace thinking message with real response
      setMessages(prev => prev.map(msg => {
        if (msg.id === thinkingId) {
          return {
            ...msg,
            content: response.text,
            isThinking: false,
            ragContext: response.contexts
          };
        }
        return msg;
      }));
    } catch (e) {
      setMessages(prev => prev.map(msg => {
        if (msg.id === thinkingId) {
          return { ...msg, content: "Sorry, something went wrong.", isThinking: false };
        }
        return msg;
      }));
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="relative h-full flex flex-col">
      <div className="absolute inset-0 z-0 bg-gradient-to-br from-slate-900 via-slate-950 to-indigo-950 animate-gradient-xy" />
      
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-8 relative z-10" ref={scrollRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`flex items-start gap-4 max-w-4xl mx-auto ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            {/* Avatar */}
            <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 border-2 ${
              msg.role === 'user' ? 'border-indigo-500 bg-indigo-900' : 'border-emerald-500 bg-emerald-900'
            }`}>
              {msg.role === 'user' ? <User className="w-5 h-5 text-indigo-300" /> : <Bot className="w-5 h-5 text-emerald-300" />}
            </div>

            {/* Message Content */}
            <div className={`flex flex-col w-full ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div className={`p-4 rounded-2xl text-sm leading-relaxed max-w-3xl ${
                msg.role === 'user' 
                  ? 'bg-slate-800 border border-slate-700 rounded-br-none' 
                  : 'bg-slate-900/80 backdrop-blur-sm border border-slate-700 rounded-bl-none'
              }`}>
                {msg.isThinking ? (
                  <div className="flex items-center gap-3 text-slate-400">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse " style={{animationDelay: '0s'}}/>
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse " style={{animationDelay: '0.2s'}}/>
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse " style={{animationDelay: '0.4s'}}/>
                  </div>
                ) : (
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                )}
              </div>
              {msg.ragContext && msg.ragContext.length > 0 && (
                <RagContextDisplay contexts={msg.ragContext} />
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="relative z-10 p-4">
        <div className="relative max-w-4xl mx-auto">
        <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-md border border-slate-700 rounded-2xl shadow-2xl" />
          <div className="relative flex items-center p-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask about AAPL performance, compare stocks, or check trends..."
              className="w-full bg-transparent text-slate-100 placeholder-slate-500 rounded-xl py-3 pl-5 pr-14 border-none focus:ring-0 outline-none"
              disabled={isProcessing}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isProcessing}
              className="absolute right-3 p-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
        <p className="text-center text-xs text-slate-500 mt-3">
          Powered by Gemini 2.5 Flash & Simulated Vector Store
        </p>
      </div>
    </div>
  );
};

const RagContextDisplay: React.FC<{ contexts: RagContext[] }> = ({ contexts }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="mt-2 w-full max-w-md">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs text-slate-500 hover:text-indigo-400 transition-colors"
      >
        {isOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <Database className="w-3 h-3" />
        Used {contexts.length} Retrieval Fragments
      </button>
      
      {isOpen && (
        <div className="mt-2 bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-lg p-3 text-xs overflow-hidden">
          <p className="font-semibold text-slate-400 mb-2">Retrieved Context (Top 3 shown):</p>
          <div className="space-y-2">
            {contexts.slice(0, 3).map((ctx, idx) => (
              <div key={idx} className="bg-slate-800/50 p-2 rounded border-l-2 border-indigo-500">
                <div className="flex justify-between text-[10px] text-indigo-300 uppercase font-mono mb-1">
                  <span>{ctx.symbol}</span>
                  <span>{ctx.date}</span>
                </div>
                <p className="text-slate-400 truncate">{ctx.summary}</p>
              </div>
            ))}
            {contexts.length > 3 && (
              <p className="text-center text-slate-600 italic">...and {contexts.length - 3} more</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatInterface;