import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Chat } from './pages/Chat';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="chat" element={<Chat />} />
          <Route path="datasets" element={<div className="p-4 text-center text-foreground/50 mt-10">Datasets page coming soon...</div>} />
          <Route path="visualizations" element={<div className="p-4 text-center text-foreground/50 mt-10">Visualizations page coming soon...</div>} />
          <Route path="models" element={<div className="p-4 text-center text-foreground/50 mt-10">ML Models page coming soon...</div>} />
          <Route path="reports" element={<div className="p-4 text-center text-foreground/50 mt-10">Reports page coming soon...</div>} />
          <Route path="*" element={<div className="p-4 text-center text-foreground/50 mt-10">404 - Page not found</div>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
