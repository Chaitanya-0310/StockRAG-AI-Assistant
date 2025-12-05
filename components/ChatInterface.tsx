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
    <div className="flex flex-col h-[calc(100vh-100px)] max-w-5xl mx-auto bg-slate-900/50 backdrop-blur rounded-2xl border border-slate-800 overflow-hidden shadow-2xl">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6" ref={scrollRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            {/* Avatar */}
            <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${
              msg.role === 'user' ? 'bg-indigo-600' : 'bg-emerald-600'
            }`}>
              {msg.role === 'user' ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-white" />}
            </div>

            {/* Message Bubble */}
            <div className={`flex flex-col max-w-[80%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div className={`p-4 rounded-2xl text-sm leading-relaxed ${
                msg.role === 'user' 
                  ? 'bg-indigo-600 text-white rounded-tr-none' 
                  : 'bg-slate-800 text-slate-200 rounded-tl-none border border-slate-700'
              }`}>
                {msg.isThinking ? (
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 animate-spin" />
                    <span className="animate-pulse">Retrieving relevant documents...</span>
                  </div>
                ) : (
                  <div className="prose prose-invert prose-sm">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                )}
              </div>

              {/* RAG Context Visualization (Collapsible) */}
              {msg.ragContext && msg.ragContext.length > 0 && (
                <RagContextDisplay contexts={msg.ragContext} />
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-slate-950 border-t border-slate-800">
        <div className="relative flex items-center max-w-4xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about AAPL performance, compare stocks, or check trends..."
            className="w-full bg-slate-900 text-slate-100 placeholder-slate-500 rounded-xl py-4 pl-6 pr-14 border border-slate-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition-all shadow-inner"
            disabled={isProcessing}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isProcessing}
            className="absolute right-2 p-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        <p className="text-center text-xs text-slate-500 mt-2">
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
        <div className="mt-2 bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs overflow-hidden">
          <p className="font-semibold text-slate-400 mb-2">Retrieved Context (Top 3 shown):</p>
          <div className="space-y-2">
            {contexts.slice(0, 3).map((ctx, idx) => (
              <div key={idx} className="bg-slate-900 p-2 rounded border-l-2 border-indigo-500">
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