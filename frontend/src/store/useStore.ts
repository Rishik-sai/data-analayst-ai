import { create } from 'zustand';

interface User {
  id: string;
  email: string;
  username: string;
  role: string;
}

interface Dataset {
  id: string;
  name: string;
  status: string;
}

interface AppState {
  user: User | null;
  datasets: Dataset[];
  activeDatasetId: string | null;
  setUser: (user: User | null) => void;
  setDatasets: (datasets: Dataset[]) => void;
  setActiveDatasetId: (id: string | null) => void;
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  theme: 'light' | 'dark' | 'system';
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
}

// Helper to apply theme to document
const applyTheme = (theme: 'light' | 'dark' | 'system') => {
  const root = window.document.documentElement;
  root.classList.remove('light', 'dark');
  
  if (theme === 'system') {
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    root.classList.add(systemTheme);
  } else {
    root.classList.add(theme);
  }
};

// Initialize theme from localStorage or system preference
const savedTheme = (localStorage.getItem('app-theme') as 'light' | 'dark' | 'system') || 'system';
applyTheme(savedTheme);

export const useStore = create<AppState>((set) => ({
  user: null,
  datasets: [],
  activeDatasetId: '1baf1412-170c-4101-ab96-913e7ff9f096',
  setUser: (user) => set({ user }),
  setDatasets: (datasets) => set({ datasets }),
  setActiveDatasetId: (id) => set({ activeDatasetId: id }),
  isSidebarOpen: true,
  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
  theme: savedTheme,
  setTheme: (theme) => {
    localStorage.setItem('app-theme', theme);
    applyTheme(theme);
    set({ theme });
  },
}));
