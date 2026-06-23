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
}

export const useStore = create<AppState>((set) => ({
  user: null,
  datasets: [],
  activeDatasetId: null,
  setUser: (user) => set({ user }),
  setDatasets: (datasets) => set({ datasets }),
  setActiveDatasetId: (id) => set({ activeDatasetId: id }),
  isSidebarOpen: true,
  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
}));
