import React from 'react';
import { UploadCloud, FileText, Activity, BrainCircuit } from 'lucide-react';

export function Dashboard() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-foreground/60 mt-1">Overview of your data analysis workspace.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { title: 'Total Datasets', value: '12', icon: FileText, color: 'text-blue-400', bg: 'bg-blue-400/10' },
          { title: 'Analyses Run', value: '48', icon: Activity, color: 'text-green-400', bg: 'bg-green-400/10' },
          { title: 'Models Trained', value: '7', icon: BrainCircuit, color: 'text-purple-400', bg: 'bg-purple-400/10' },
          { title: 'Storage Used', value: '450 MB', icon: UploadCloud, color: 'text-orange-400', bg: 'bg-orange-400/10' },
        ].map((stat, i) => (
          <div key={i} className="bg-card border border-border p-5 rounded-xl flex items-center gap-4">
            <div className={`p-3 rounded-lg ${stat.bg} ${stat.color}`}>
              <stat.icon className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm text-foreground/60 font-medium">{stat.title}</p>
              <h3 className="text-2xl font-bold mt-0.5">{stat.value}</h3>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-card border border-border rounded-xl p-6 min-h-[400px]">
          <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
          <div className="flex flex-col items-center justify-center h-full text-foreground/40">
            <Activity className="w-12 h-12 mb-3 opacity-20" />
            <p>No recent activity. Upload a dataset to get started.</p>
          </div>
        </div>
        <div className="bg-card border border-border rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
          <div className="space-y-3">
            <button className="w-full bg-primary hover:bg-primary/90 text-primary-foreground py-2.5 rounded-lg font-medium transition-colors flex items-center justify-center gap-2">
              <UploadCloud className="w-5 h-5" />
              Upload Dataset
            </button>
            <button className="w-full bg-secondary hover:bg-secondary/80 text-foreground py-2.5 rounded-lg border border-border transition-colors flex items-center justify-center gap-2">
              <MessageSquare className="w-5 h-5" />
              Ask AI Assistant
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Temporary import for icon
import { MessageSquare } from 'lucide-react';
