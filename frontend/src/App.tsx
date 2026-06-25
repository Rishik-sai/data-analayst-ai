import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Chat } from './pages/Chat';
import { Datasets } from './pages/Datasets';
import { Visualizations } from './pages/Visualizations';
import { Reports } from './pages/Reports';
import { Settings } from './pages/Settings';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="chat" element={<Chat />} />
          <Route path="datasets" element={<Datasets />} />
          <Route path="visualizations" element={<Visualizations />} />
          <Route path="models" element={<div className="p-4 text-center text-foreground/50 mt-10">ML Models page coming soon...</div>} />
          <Route path="reports" element={<Reports />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<div className="p-4 text-center text-foreground/50 mt-10">404 - Page not found</div>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
