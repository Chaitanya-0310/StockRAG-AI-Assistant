import React, { useState } from 'react';
import { AppTab } from './types';
import ChatInterface from './components/ChatInterface';
import StockDashboard from './components/StockDashboard';
import PipelineVisualizer from './components/PipelineVisualizer';
import { getAllTickers } from './services/mockDataService';
import { LayoutDashboard, MessageSquare, DatabaseZap, Terminal } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState<AppTab>(AppTab.CHAT);
  const tickerCount = getAllTickers().length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans flex">
      {/* Sidebar */}
      <aside className="w-20 lg:w-64 border-r border-slate-800 bg-slate-900 flex flex-col shrink-0 transition-all">
        <div className="h-20 flex items-center justify-center lg:justify-start lg:px-6 border-b border-slate-800">
          <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <Terminal className="text-white w-6 h-6" />
          </div>
          <span className="ml-3 font-bold text-xl text-white hidden lg:block">StockRAG</span>
        </div>

        <nav className="p-4 space-y-2 flex-1">
          <SidebarItem 
            icon={<MessageSquare size={20} />} 
            label="AI Analyst" 
            isActive={activeTab === AppTab.CHAT} 
            onClick={() => setActiveTab(AppTab.CHAT)} 
          />
          <SidebarItem 
            icon={<LayoutDashboard size={20} />} 
            label="Data Explorer" 
            isActive={activeTab === AppTab.DASHBOARD} 
            onClick={() => setActiveTab(AppTab.DASHBOARD)} 
          />
          <SidebarItem 
            icon={<DatabaseZap size={20} />} 
            label="ETL Pipeline" 
            isActive={activeTab === AppTab.PIPELINE} 
            onClick={() => setActiveTab(AppTab.PIPELINE)} 
          />
        </nav>

        <div className="p-4 border-t border-slate-800 hidden lg:block">
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-2">System Status</p>
            <div className="flex items-center gap-2 text-sm text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              RAG Engine Online
            </div>
            <div className="flex items-center gap-2 text-sm text-blue-400 mt-1">
              <span className="w-2 h-2 rounded-full bg-blue-500"></span>
              {tickerCount} Tickers Indexed
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        <header className="h-20 border-b border-slate-800 bg-slate-900/50 backdrop-blur flex items-center px-8 justify-between">
          <h1 className="text-2xl font-semibold text-white">
            {activeTab === AppTab.CHAT && 'AI Financial Assistant'}
            {activeTab === AppTab.DASHBOARD && 'Market Data Dashboard'}
            {activeTab === AppTab.PIPELINE && 'Pipeline Configuration'}
          </h1>
          
          <div className="hidden md:flex items-center gap-4">
            <div className="text-right">
              <p className="text-sm text-white font-medium">Demo User</p>
              <p className="text-xs text-slate-500">Premium Plan</p>
            </div>
            <div className="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
              <span className="font-bold text-indigo-400">DU</span>
            </div>
          </div>
        </header>

        <div className="flex-1 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] bg-opacity-20 relative">
          {/* Content Wrapper */}
          <div className="absolute inset-0 overflow-hidden">
             {activeTab === AppTab.CHAT && <div className="h-full p-6"><ChatInterface /></div>}
             {activeTab === AppTab.DASHBOARD && <StockDashboard />}
             {activeTab === AppTab.PIPELINE && <PipelineVisualizer />}
          </div>
        </div>
      </main>
    </div>
  );
}

// Subcomponent for cleaner code
const SidebarItem = ({ icon, label, isActive, onClick }: { icon: React.ReactNode, label: string, isActive: boolean, onClick: () => void }) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
      isActive 
        ? 'bg-indigo-600/10 text-indigo-400 border border-indigo-600/20 shadow-[0_0_15px_rgba(99,102,241,0.1)]' 
        : 'text-slate-400 hover:bg-slate-800 hover:text-white'
    }`}
  >
    <div className={`transition-transform duration-200 ${isActive ? 'scale-110' : 'group-hover:scale-110'}`}>
      {icon}
    </div>
    <span className="font-medium hidden lg:block">{label}</span>
    {isActive && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-indigo-400 hidden lg:block shadow-[0_0_8px_currentColor]" />}
  </button>
);

export default App;