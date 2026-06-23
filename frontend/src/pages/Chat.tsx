import React, { useState } from 'react';
import { Send, Bot, User, BrainCircuit } from 'lucide-react';
import clsx from 'clsx';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I am your AI Data Analyst. How can I help you explore your data today?'
    }
  ]);
  const [input, setInput] = useState('');

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const newMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages(prev => [...prev, newMsg]);
    setInput('');
    
    // Simulate AI response
    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I'm currently running in mockup mode, but when the backend is connected, I will analyze your data and respond to that."
      }]);
    }, 1000);
  };

  return (
    <div className="flex flex-col h-full bg-card border border-border rounded-xl overflow-hidden shadow-sm">
      <div className="p-4 border-b border-border bg-card/80 backdrop-blur-sm flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center text-primary">
          <BrainCircuit className="w-6 h-6" />
        </div>
        <div>
          <h2 className="font-semibold">AI Data Analyst</h2>
          <p className="text-xs text-foreground/60 flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500 inline-block animate-pulse"></span>
            Online and ready
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.map((msg) => (
          <div key={msg.id} className={clsx("flex gap-4 max-w-[85%]", msg.role === 'user' ? "ml-auto flex-row-reverse" : "")}>
            <div className={clsx(
              "w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center mt-1",
              msg.role === 'user' ? "bg-secondary text-foreground" : "bg-primary text-primary-foreground"
            )}>
              {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
            </div>
            <div className={clsx(
              "p-4 rounded-2xl",
              msg.role === 'user' 
                ? "bg-primary/20 text-foreground rounded-tr-sm" 
                : "bg-secondary border border-border text-foreground rounded-tl-sm"
            )}>
              <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="p-4 border-t border-border bg-card/50">
        <form onSubmit={handleSend} className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your data, request charts, or build models..."
            className="w-full bg-background border border-border rounded-full pl-6 pr-14 py-4 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all text-foreground"
          />
          <button 
            type="submit"
            disabled={!input.trim()}
            className="absolute right-2 p-2.5 bg-primary hover:bg-primary/90 text-primary-foreground rounded-full disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5 ml-0.5" />
          </button>
        </form>
      </div>
    </div>
  );
}
