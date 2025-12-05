import React, { useMemo } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ComposedChart, Bar, Cell, Line
} from 'recharts';
import { StockDataPoint } from '../types';

interface StockChartProps {
  data: StockDataPoint[];
  color?: string;
  type?: 'area' | 'candle';
  showSMA?: boolean;
  className?: string;
}

// Custom Tooltip to display OHLC data clearly
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;

    return (
      <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg shadow-xl text-xs backdrop-blur-md bg-opacity-95 z-50">
        <p className="text-slate-400 mb-2 font-semibold border-b border-slate-800 pb-1">
          {new Date(label).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}
        </p>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1">
          <div className="text-slate-500">Open</div>
          <div className="text-slate-200 text-right font-mono">${data.open.toFixed(2)}</div>

          <div className="text-slate-500">High</div>
          <div className="text-slate-200 text-right font-mono">${data.high.toFixed(2)}</div>

          <div className="text-slate-500">Low</div>
          <div className="text-slate-200 text-right font-mono">${data.low.toFixed(2)}</div>

          <div className="text-slate-500 font-medium">Close</div>
          <div className={`text-right font-mono font-bold ${data.close >= data.open ? 'text-emerald-400' : 'text-rose-400'}`}>
            ${data.close.toFixed(2)}
          </div>

          {data.sma && (
            <>
              <div className="text-amber-400 mt-1 pt-1 border-t border-slate-800">SMA (20)</div>
              <div className="text-amber-400 text-right font-mono mt-1 pt-1 border-t border-slate-800">
                ${data.sma.toFixed(2)}
              </div>
            </>
          )}

          <div className="text-slate-600 mt-1">Vol</div>
          <div className="text-slate-500 text-right font-mono mt-1">
            {(data.volume / 1000000).toFixed(2)}M
          </div>
        </div>
      </div>
    );
  }
  return null;
};

const StockChart: React.FC<StockChartProps> = ({
  data,
  color = "#6366f1",
  type = 'area',
  showSMA = false,
  className = "h-[300px]"
}) => {
  const rawId = React.useId();
  const gradientId = rawId.replace(/:/g, "");

  // Pre-process data to calculate SMA and formatting for Candle
  const processedData = useMemo(() => {
    return data.map((point, index, array) => {
      // Calculate SMA 20
      let sma = null;
      if (index >= 19) {
        const slice = array.slice(index - 19, index + 1);
        const sum = slice.reduce((acc, curr) => acc + curr.close, 0);
        sma = sum / 20;
      }

      // Prepare Candle Data
      const isUp = point.close >= point.open;
      const candleColor = isUp ? '#10b981' : '#f43f5e'; // Emerald-500 : Rose-500

      return {
        ...point,
        sma,
        candleColor,
        // Wick ranges from Low to High
        wick: [point.low, point.high],
        // Body ranges from Min(Open,Close) to Max(Open,Close)
        body: [Math.min(point.open, point.close), Math.max(point.open, point.close)]
      };
    });
  }, [data]);

  if (!data || data.length === 0) {
    return (
      <div className={`w-full ${className} flex items-center justify-center text-slate-500`}>
        No Data Available
      </div>
    );
  }

  return (
    <div className={`w-full ${className}`} style={{ minHeight: 300 }}>
      <ResponsiveContainer width="100%" height="100%" minHeight={300}>
        <ComposedChart
          data={processedData}
          margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />

          <XAxis
            dataKey="date"
            stroke="#64748b"
            tick={{ fontSize: 11 }}
            tickFormatter={(str) => {
              const d = new Date(str);
              return `${d.getMonth() + 1}/${d.getDate()}`;
            }}
            minTickGap={30}
          />

          <YAxis
            domain={['auto', 'auto']}
            stroke="#64748b"
            tick={{ fontSize: 11 }}
            tickFormatter={(val) => `$${val.toFixed(0)}`}
            width={40}
          />

          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#475569', strokeDasharray: '4 4' }} />

          {/* AREA CHART MODE */}
          {type === 'area' && (
            <Area
              type="monotone"
              dataKey="close"
              stroke={color}
              strokeWidth={2}
              fillOpacity={1}
              fill={`url(#${gradientId})`}
              activeDot={{ r: 4, strokeWidth: 0 }}
            />
          )}

          {/* CANDLESTICK MODE */}
          {type === 'candle' && (
            <>
              {/* Wick (Thin bar) */}
              <Bar dataKey="wick" barSize={1} isAnimationActive={false}>
                {processedData.map((entry, index) => (
                  <Cell key={`wick-${index}`} fill={entry.candleColor} />
                ))}
              </Bar>

              {/* Body (Thicker bar) */}
              <Bar dataKey="body" barSize={8} isAnimationActive={false}>
                {processedData.map((entry, index) => (
                  <Cell key={`body-${index}`} fill={entry.candleColor} />
                ))}
              </Bar>
            </>
          )}

          {/* SMA OVERLAY */}
          {showSMA && (
            <Line
              type="monotone"
              dataKey="sma"
              stroke="#fbbf24" // Amber-400
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#fbbf24' }}
            />
          )}

        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default StockChart;