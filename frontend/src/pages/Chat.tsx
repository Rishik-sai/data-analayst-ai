import { fetchWithAuth, API_BASE_URL } from '../utils/apiClient';
import React, { useEffect, useRef, useState } from 'react';
import { Send, Bot, User, BrainCircuit, Loader2, Database, Target, Sparkles, Trophy, Trash2, Star, Network } from 'lucide-react';
import clsx from 'clsx';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
import { useStore } from '../store/useStore';

interface ChartSpec {
  type: 'bar' | 'heatmap';
  title: string;
  data?: { name: string; value: number }[];
  format?: 'percent' | 'number';
  layout?: 'vertical';
  highlight?: string;
  columns?: string[];
  values?: number[][];
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  charts?: ChartSpec[];
  isError?: boolean;
}

interface DatasetOption {
  id: string;
  name: string;
  row_count?: number;
  column_count?: number;
  columns_metadata?: { name: string }[];
}

const SUGGESTIONS = [
  { icon: Trophy, label: 'Which model gives the highest accuracy?' },
  { icon: Trash2, label: 'What are the unnecessary columns?' },
  { icon: Star, label: 'What are the most important columns?' },
  { icon: Network, label: 'What are the relationships between columns?' },
  { icon: Sparkles, label: 'Analyze this dataset' },
];

const welcome = (dataset?: DatasetOption): Message => ({
  id: 'welcome',
  role: 'assistant',
  content: dataset
    ? `You're chatting about **${dataset.name}**${dataset.row_count ? ` (${dataset.row_count.toLocaleString()} rows × ${dataset.column_count} columns)` : ''}.\n\nAsk me which model gets the highest accuracy, which columns are unnecessary or most important, or how the columns relate — or pick a suggestion below.`
    : 'Hello! I am your AI Data Analyst. Select or upload a dataset to get model recommendations, column importance and relationships — or ask me anything about data.',
});

const fmt = (v: number, format?: string) => (format === 'percent' ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));

function BarChartCard({ chart }: { chart: ChartSpec }) {
  const data = chart.data || [];
  const vertical = chart.layout === 'vertical';
  const height = vertical ? Math.max(200, data.length * 28 + 40) : 260;
  return (
    <div className="bg-background border border-border rounded-xl p-4">
      <h4 className="text-sm font-semibold mb-3">{chart.title}</h4>
      <ResponsiveContainer width="100%" height={height}>
        {vertical ? (
          <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
            <XAxis type="number" tickFormatter={v => fmt(v, chart.format)} fontSize={11} />
            <YAxis type="category" dataKey="name" width={120} fontSize={11} />
            <Tooltip formatter={(v) => fmt(Number(v), chart.format)} />
            <Bar dataKey="value" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
          </BarChart>
        ) : (
          <BarChart data={data} margin={{ bottom: 50 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
            <XAxis dataKey="name" angle={-30} textAnchor="end" interval={0} fontSize={11} />
            <YAxis tickFormatter={v => fmt(v, chart.format)} fontSize={11} domain={chart.format === 'percent' ? [0, 1] : ['auto', 'auto']} />
            <Tooltip formatter={(v) => fmt(Number(v), chart.format)} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.map(d => (
                <Cell key={d.name} fill="hsl(var(--primary))" fillOpacity={d.name === chart.highlight ? 1 : 0.45} />
              ))}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

function HeatmapCard({ chart }: { chart: ChartSpec }) {
  const cols = chart.columns || [];
  const values = chart.values || [];
  return (
    <div className="bg-background border border-border rounded-xl p-4 overflow-x-auto">
      <h4 className="text-sm font-semibold mb-1">{chart.title}</h4>
      <p className="text-xs text-foreground/60 mb-3">Darker = stronger relationship. Blue = positive, red = negative (numeric pairs).</p>
      <table className="text-[11px] border-separate" style={{ borderSpacing: 2 }}>
        <thead>
          <tr>
            <th />
            {cols.map(c => (
              <th key={c} className="font-medium text-foreground/70 align-bottom h-24">
                <div className="[writing-mode:vertical-rl] rotate-180 whitespace-nowrap mx-auto">{c}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cols.map((row, i) => (
            <tr key={row}>
              <th className="font-medium text-foreground/70 text-right pr-2 whitespace-nowrap">{row}</th>
              {cols.map((_, j) => {
                const v = values[i]?.[j] ?? 0;
                const color = v >= 0 ? '59,130,246' : '239,68,68';
                return (
                  <td
                    key={j}
                    title={`${row} × ${cols[j]}: ${v.toFixed(2)}`}
                    className="w-9 h-9 text-center rounded"
                    style={{ background: `rgba(${color},${Math.min(1, Math.abs(v)) * 0.85 + 0.05})`, color: Math.abs(v) > 0.5 ? 'white' : undefined }}
                  >
                    {i === j ? '' : v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Chat() {
  const activeDatasetId = useStore(state => state.activeDatasetId);
  const setActiveDatasetId = useStore(state => state.setActiveDatasetId);
  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [messages, setMessages] = useState<Message[]>([welcome()]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const activeDataset = datasets.find(d => d.id === activeDatasetId);
  const columns = activeDataset?.columns_metadata?.map(c => c.name) || [];

  useEffect(() => {
    fetchWithAuth(`${API_BASE_URL}/api/datasets`)
      .then(res => (res.ok ? res.json() : []))
      .then(setDatasets)
      .catch(() => setDatasets([]));
  }, []);

  // New dataset → fresh conversation
  useEffect(() => {
    setMessages([welcome(activeDataset)]);
    setSessionId(null);
    setTargetColumn('');
  }, [activeDatasetId, activeDataset?.name]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, isLoading]);

  const send = async (text: string) => {
    if (!text.trim() || isLoading) return;
    setMessages(prev => [...prev, { id: Date.now().toString(), role: 'user', content: text }]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetchWithAuth(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          dataset_id: activeDatasetId,
          session_id: sessionId,
          target_column: targetColumn || null,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to get response from AI');
      }

      const data = await response.json();
      setSessionId(data.session_id);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.response,
        charts: data.charts || undefined,
      }]);
    } catch (error: any) {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Error: ${error.message || "I couldn't connect to the backend."}`,
        isError: true,
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    send(input);
  };

  return (
    <div className="flex flex-col h-full bg-card border border-border rounded-xl overflow-hidden shadow-sm">
      <div className="p-4 border-b border-border bg-card/80 backdrop-blur-sm flex flex-wrap items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center text-primary">
          <BrainCircuit className="w-6 h-6" />
        </div>
        <div className="mr-auto">
          <h2 className="font-semibold">AI Data Analyst</h2>
          <p className="text-xs text-foreground/60 flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500 inline-block animate-pulse"></span>
            Online and ready
          </p>
        </div>

        <label className="flex items-center gap-2 text-sm">
          <Database className="w-4 h-4 text-foreground/60" />
          <select
            value={activeDatasetId || ''}
            onChange={e => setActiveDatasetId(e.target.value || null)}
            className="bg-background border border-border rounded-lg px-2 py-1.5 max-w-[200px] focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="">No dataset</option>
            {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
        </label>

        {activeDataset && (
          <label className="flex items-center gap-2 text-sm" title="Column to predict. Auto picks the most likely target.">
            <Target className="w-4 h-4 text-foreground/60" />
            <select
              value={targetColumn}
              onChange={e => setTargetColumn(e.target.value)}
              className="bg-background border border-border rounded-lg px-2 py-1.5 max-w-[180px] focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              <option value="">Target: auto</option>
              {columns.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
        )}
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.map((msg) => (
          <div key={msg.id} className={clsx("flex gap-4", msg.role === 'user' ? "ml-auto flex-row-reverse max-w-[85%]" : "max-w-[95%]")}>
            <div className={clsx(
              "w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center mt-1",
              msg.role === 'user' ? "bg-secondary text-foreground" : "bg-primary text-primary-foreground"
            )}>
              {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
            </div>
            <div className={clsx(
              "p-4 rounded-2xl min-w-0",
              msg.role === 'user'
                ? "bg-primary/20 text-foreground rounded-tr-sm"
                : "bg-secondary border border-border text-foreground rounded-tl-sm",
              msg.isError && "border-red-500/50 text-red-500"
            )}>
              {msg.role === 'user' ? (
                <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
              ) : (
                <div className="prose prose-sm dark:prose-invert max-w-none prose-table:text-xs prose-th:px-2 prose-td:px-2 prose-table:block prose-table:overflow-x-auto">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                </div>
              )}
              {msg.charts && msg.charts.length > 0 && (
                <div className="grid gap-4 mt-4 xl:grid-cols-2">
                  {msg.charts.map((c, i) => c.type === 'heatmap'
                    ? <div key={i} className="xl:col-span-2"><HeatmapCard chart={c} /></div>
                    : <BarChartCard key={i} chart={c} />)}
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex gap-4 max-w-[85%]">
            <div className="w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center mt-1 bg-primary text-primary-foreground">
              <Bot className="w-5 h-5" />
            </div>
            <div className="p-4 rounded-2xl bg-secondary border border-border text-foreground rounded-tl-sm flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-primary" />
              <span className="text-sm text-foreground/70">Analyzing… (training and comparing models can take a few seconds)</span>
            </div>
          </div>
        )}
      </div>

      <div className="p-4 border-t border-border bg-card/50 space-y-3">
        {activeDataset && (
          <div className="flex flex-wrap gap-2">
            {SUGGESTIONS.map(({ icon: Icon, label }) => (
              <button
                key={label}
                onClick={() => send(label)}
                disabled={isLoading}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border border-border bg-background hover:border-primary hover:text-primary disabled:opacity-50 transition-colors"
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </button>
            ))}
          </div>
        )}
        <form onSubmit={handleSend} className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={activeDataset ? `Ask about ${activeDataset.name}… e.g. "best model to predict ${columns[columns.length - 1] || 'price'}"` : 'Ask about your data, request charts, or build models...'}
            className="w-full bg-background border border-border rounded-full pl-6 pr-14 py-4 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all text-foreground"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="absolute right-2 p-2.5 bg-primary hover:bg-primary/90 text-primary-foreground rounded-full disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5 ml-0.5" />
          </button>
        </form>
      </div>
    </div>
  );
}
