import React from 'react';
import { Sidebar } from './Sidebar';
import { Outlet } from 'react-router-dom';

export function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <header className="h-16 border-b border-border flex items-center justify-between px-6 bg-card/50 backdrop-blur-sm z-10">
          <div className="font-medium">Welcome to DataMind AI</div>
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
              U
            </div>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-6 relative">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
