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
  token: string | null;
  datasets: Dataset[];
  activeDatasetId: string | null;
  setUser: (user: User | null) => void;
  setAuth: (user: User | null, token: string | null, rememberMe?: boolean) => void;
  logout: () => void;
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

// Helper to read initial token:
// Checks sessionStorage first (per-session), then localStorage ONLY if user explicitly chose "remember-me"
const getInitialToken = (): string | null => {
  const sessionTok = sessionStorage.getItem('datamind-token');
  if (sessionTok) return sessionTok;
  return localStorage.getItem('datamind-token');
};

export const useStore = create<AppState>((set) => ({
  user: null,
  token: getInitialToken(),
  datasets: [],
  activeDatasetId: null,
  setUser: (user) => set({ user }),
  setAuth: (user, token, rememberMe = false) => {
    if (token) {
      if (rememberMe) {
        localStorage.setItem('datamind-token', token);
        sessionStorage.removeItem('datamind-token');
      } else {
        sessionStorage.setItem('datamind-token', token);
        localStorage.removeItem('datamind-token');
      }
    } else {
      localStorage.removeItem('datamind-token');
      sessionStorage.removeItem('datamind-token');
    }
    set({ user, token });
  },
  logout: () => {
    localStorage.removeItem('datamind-token');
    sessionStorage.removeItem('datamind-token');
    set({ user: null, token: null, datasets: [], activeDatasetId: null });
  },
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
