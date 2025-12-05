import React, { useState } from 'react';
import { PipelineStep } from '../types';
import { 
  CheckCircle, Loader2, CircleDashed, Play, RefreshCw, 
  Code, Layout, Database, FileJson, Server 
} from 'lucide-react';

const PYTHON_AIRFLOW_CODE = `from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
import google.generativeai as genai

# Database Connection
db_engine = create_engine('postgresql://user:pass@localhost:5432/stock_db')

def extract_stock_data(**context):
    """Extracts data for 20+ tickers using generic Stock API"""
    tickers = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ', 'JPM']
    data_frames = []
    for t in tickers:
        # Fetching last 1 day of data
        df = yf.download(t, period="1d")
        df['Symbol'] = t
        data_frames.append(df)
    
    combined_df = pd.concat(data_frames)
    combined_df.to_csv('/tmp/raw_stock_data.csv')

def clean_and_transform(**context):
    """Cleans data using Pandas"""
    df = pd.read_csv('/tmp/raw_stock_data.csv')
    
    # 1. Handle Missing Values
    df.fillna(method='ffill', inplace=True)
    
    # 2. Calculate Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # 3. Save Structured Data to PostgreSQL
    df.to_sql('stock_prices', db_engine, if_exists='append', index=False)

def generate_embeddings(**context):
    """Generates Vector Embeddings for RAG"""
    df = pd.read_sql('SELECT * FROM stock_prices WHERE created_at > NOW() - interval 1 day', db_engine)
    
    for index, row in df.iterrows():
        summary = f"{row['Symbol']} closed at {row['Close']} on {row['Date']}."
        
        # Call Gemini API for Embedding
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=summary,
            task_type="retrieval_document"
        )
        
        # Store in pgvector table
        save_vector_to_db(row['Symbol'], summary, embedding['embedding'])

# Define Airflow DAG
with DAG(
    'stock_rag_pipeline',
    default_args={'retries': 1},
    schedule_interval='0 16 * * 1-5', # Market Close (Mon-Fri)
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    t1 = PythonOperator(
        task_id='extract_market_data',
        python_callable=extract_stock_data
    )

    t2 = PythonOperator(
        task_id='clean_transform_load',
        python_callable=clean_and_transform
    )

    t3 = PythonOperator(
        task_id='generate_vector_embeddings',
        python_callable=generate_embeddings
    )

    t1 >> t2 >> t3
`;

const PipelineVisualizer: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'graph' | 'code'>('graph');
  const [isRunning, setIsRunning] = useState(false);
  const [executionTime, setExecutionTime] = useState<string>('00:00:00');
  
  const [tasks, setTasks] = useState([
    { id: 'extract', label: 'extract_market_data', status: 'success', duration: '4s', operator: 'PythonOperator' },
    { id: 'transform', label: 'clean_transform_load', status: 'success', duration: '2s', operator: 'PythonOperator' },
    { id: 'vector', label: 'generate_vector_embeddings', status: 'success', duration: '12s', operator: 'PythonOperator' },
  ]);

  const runPipeline = () => {
    if (isRunning) return;
    setIsRunning(true);
    setTasks(t => t.map(task => ({ ...task, status: 'queued' })));

    // Simulation sequence
    let step = 0;
    const interval = setInterval(() => {
      if (step > 2) {
        clearInterval(interval);
        setIsRunning(false);
        return;
      }

      setTasks(prev => prev.map((t, i) => {
        if (i === step) return { ...t, status: 'running' };
        if (i === step - 1) return { ...t, status: 'success' };
        return t;
      }));

      if (step === 3) { // cleanup last step
         setTasks(prev => prev.map(t => ({ ...t, status: 'success' })));
      }

      step++;
    }, 1500);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto h-[calc(100vh-120px)] flex flex-col">
      {/* Header */}
      <div className="mb-6 flex justify-between items-center bg-slate-900/50 p-6 rounded-xl border border-slate-800">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h2 className="text-2xl font-bold text-white">DAG: stock_rag_pipeline</h2>
            <span className="bg-emerald-500/10 text-emerald-400 text-xs px-2 py-1 rounded-full border border-emerald-500/20 font-mono">active</span>
          </div>
          <p className="text-slate-400 text-sm">Schedule: <span className="font-mono text-slate-300">0 16 * * 1-5</span> (Market Close)</p>
        </div>
        
        <div className="flex gap-3">
          <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
            <button 
              onClick={() => setActiveTab('graph')}
              className={`px-4 py-2 rounded-md text-sm font-medium flex items-center gap-2 transition-all ${activeTab === 'graph' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}
            >
              <Layout className="w-4 h-4" /> Graph
            </button>
            <button 
              onClick={() => setActiveTab('code')}
              className={`px-4 py-2 rounded-md text-sm font-medium flex items-center gap-2 transition-all ${activeTab === 'code' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}
            >
              <Code className="w-4 h-4" /> Code
            </button>
          </div>

          <button
            onClick={runPipeline}
            disabled={isRunning}
            className={`px-6 py-2 rounded-lg font-bold flex items-center gap-2 transition-all ${
              isRunning 
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-500/20'
            }`}
          >
            {isRunning ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
            Trigger DAG
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 bg-slate-900 rounded-xl border border-slate-800 overflow-hidden relative">
        
        {activeTab === 'graph' && (
          <div className="h-full w-full p-8 flex items-center justify-center bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:16px_16px]">
            <div className="flex items-center gap-8 relative">
              {/* Connecting Line */}
              <div className="absolute top-1/2 left-0 w-full h-1 bg-slate-700 -z-10 -translate-y-1/2"></div>

              {tasks.map((task) => (
                <div key={task.id} className={`
                  w-64 p-4 rounded-lg border-2 bg-slate-950 transition-all duration-300
                  ${task.status === 'running' ? 'border-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.3)]' : 
                    task.status === 'success' ? 'border-emerald-500' : 'border-slate-700'}
                `}>
                  <div className="flex justify-between items-start mb-3">
                    <div className={`p-2 rounded ${task.status === 'running' ? 'bg-blue-500/20 text-blue-400' : task.status === 'success' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-500'}`}>
                       {task.status === 'running' ? <Loader2 className="w-5 h-5 animate-spin" /> : 
                        task.status === 'success' ? <CheckCircle className="w-5 h-5" /> : 
                        <CircleDashed className="w-5 h-5" />}
                    </div>
                    <span className="text-xs font-mono text-slate-500">{task.duration}</span>
                  </div>
                  <h4 className="font-bold text-slate-200 text-sm mb-1">{task.label}</h4>
                  <p className="text-xs text-slate-500 font-mono">{task.operator}</p>
                </div>
              ))}
            </div>

            {/* Architecture Explainer Overlay */}
            <div className="absolute bottom-6 left-6 right-6 flex justify-between text-xs text-slate-500 font-mono">
              <div className="flex flex-col items-center gap-2">
                <Server className="w-8 h-8 text-slate-600" />
                <span>Stock API</span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <Database className="w-8 h-8 text-blue-500" />
                <span>PostgreSQL (Raw)</span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <Database className="w-8 h-8 text-indigo-500" />
                <span>pgvector (Embeddings)</span>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'code' && (
          <div className="h-full w-full overflow-hidden flex flex-col">
            <div className="bg-slate-950 p-3 border-b border-slate-800 flex justify-between items-center">
              <span className="text-xs text-slate-400 font-mono">/dags/stock_pipeline.py</span>
              <button className="text-xs text-indigo-400 hover:text-indigo-300">Copy to Clipboard</button>
            </div>
            <div className="flex-1 overflow-auto p-4 bg-[#0d1117]">
              <pre className="font-mono text-sm leading-relaxed text-slate-300">
                <code>
                  {PYTHON_AIRFLOW_CODE.split('\n').map((line, i) => (
                    <div key={i} className="table-row">
                      <span className="table-cell text-right pr-4 text-slate-700 select-none w-8">{i + 1}</span>
                      <span className="table-cell">
                        {line
                          .replace(/import|from|as/g, '<span class="text-rose-400">$&</span>')
                          .replace(/def|class|return/g, '<span class="text-blue-400">$&</span>')
                          .replace(/'.*?'/g, '<span class="text-emerald-400">$&</span>')
                          .replace(/#.*/g, '<span class="text-slate-500">$&</span>')
                        }
                      </span>
                    </div>
                  )).reduce((acc: any, curr) => {
                     // Simple dangerousHTML workaround for this demo visualization
                     return <div dangerouslySetInnerHTML={{__html: curr.props.children[1].props.children}} className="whitespace-pre pl-8 relative" />
                  }, [])}
                  {/* Fallback plain text render if dangerous HTML is risky, but for this mock string it's fine */}
                  {PYTHON_AIRFLOW_CODE}
                </code>
              </pre>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default PipelineVisualizer;