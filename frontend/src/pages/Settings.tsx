import React, { useEffect, useState } from 'react';
import { useStore } from '../store/useStore';
import { Settings as SettingsIcon, User, Moon, Sun, Monitor, Bell, Shield, Mail } from 'lucide-react';

interface UserProfile {
  id: string;
  email: string;
  username: string;
  full_name: string | null;
  role: string;
  created_at: string;
}

export function Settings() {
  const { theme, setTheme } = useStore();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Mock states for UI
  const [emailNotifs, setEmailNotifs] = useState(true);
  const [dataSharing, setDataSharing] = useState(false);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/users/me');
        if (res.ok) {
          const data = await res.json();
          setProfile(data);
        }
      } catch (err) {
        console.error("Failed to fetch profile", err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchProfile();
  }, []);

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      <div>
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
          <SettingsIcon className="w-8 h-8 text-primary" />
          Settings
        </h1>
        <p className="text-foreground/60 mt-1">Manage your account and application preferences.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-1 space-y-2">
          <div className="font-semibold text-lg flex items-center gap-2 mb-4">
            <User className="w-5 h-5 text-primary" />
            Account Details
          </div>
          <p className="text-sm text-foreground/60">
            View your personal information and current role within the system.
          </p>
        </div>

        <div className="md:col-span-2 bg-card border border-border rounded-xl p-6">
          {isLoading ? (
            <div className="flex justify-center p-6">
              <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
            </div>
          ) : profile ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium mb-1 text-foreground/60">Username</label>
                  <div className="bg-secondary/50 border border-border rounded-lg px-4 py-2.5 text-foreground font-medium">
                    {profile.username}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1 text-foreground/60">Email Address</label>
                  <div className="bg-secondary/50 border border-border rounded-lg px-4 py-2.5 text-foreground font-medium flex items-center gap-2">
                    <Mail className="w-4 h-4 text-foreground/40" />
                    {profile.email}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1 text-foreground/60">Full Name</label>
                  <div className="bg-secondary/50 border border-border rounded-lg px-4 py-2.5 text-foreground font-medium">
                    {profile.full_name || 'Not provided'}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1 text-foreground/60">Role</label>
                  <div className="bg-secondary/50 border border-border rounded-lg px-4 py-2.5 text-foreground font-medium capitalize">
                    {profile.role}
                  </div>
                </div>
              </div>
              <div className="pt-4 border-t border-border">
                <button className="text-sm text-primary hover:underline font-medium">Change Password</button>
              </div>
            </div>
          ) : (
            <p className="text-foreground/50">Failed to load profile.</p>
          )}
        </div>
      </div>

      <div className="w-full h-px bg-border my-8"></div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-1 space-y-2">
          <div className="font-semibold text-lg flex items-center gap-2 mb-4">
            <Monitor className="w-5 h-5 text-primary" />
            Appearance
          </div>
          <p className="text-sm text-foreground/60">
            Customize the look and feel of your workspace.
          </p>
        </div>

        <div className="md:col-span-2 bg-card border border-border rounded-xl p-6">
          <h3 className="text-sm font-medium mb-4 text-foreground/60">Theme Preference</h3>
          <div className="grid grid-cols-3 gap-4">
            <button
              onClick={() => setTheme('light')}
              className={`flex flex-col items-center justify-center p-4 rounded-xl border-2 transition-all ${
                theme === 'light' ? 'border-primary bg-primary/5 text-primary' : 'border-border bg-card hover:bg-secondary/50 text-foreground/70'
              }`}
            >
              <Sun className="w-6 h-6 mb-2" />
              <span className="font-medium text-sm">Light</span>
            </button>
            <button
              onClick={() => setTheme('dark')}
              className={`flex flex-col items-center justify-center p-4 rounded-xl border-2 transition-all ${
                theme === 'dark' ? 'border-primary bg-primary/5 text-primary' : 'border-border bg-card hover:bg-secondary/50 text-foreground/70'
              }`}
            >
              <Moon className="w-6 h-6 mb-2" />
              <span className="font-medium text-sm">Dark</span>
            </button>
            <button
              onClick={() => setTheme('system')}
              className={`flex flex-col items-center justify-center p-4 rounded-xl border-2 transition-all ${
                theme === 'system' ? 'border-primary bg-primary/5 text-primary' : 'border-border bg-card hover:bg-secondary/50 text-foreground/70'
              }`}
            >
              <Monitor className="w-6 h-6 mb-2" />
              <span className="font-medium text-sm">System</span>
            </button>
          </div>
        </div>
      </div>

      <div className="w-full h-px bg-border my-8"></div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-1 space-y-2">
          <div className="font-semibold text-lg flex items-center gap-2 mb-4">
            <Bell className="w-5 h-5 text-primary" />
            Notifications & Privacy
          </div>
          <p className="text-sm text-foreground/60">
            Manage your email alerts and data sharing settings.
          </p>
        </div>

        <div className="md:col-span-2 bg-card border border-border rounded-xl p-6 space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-base font-medium">Email Notifications</label>
              <p className="text-sm text-foreground/60">Receive alerts when long-running analyses complete.</p>
            </div>
            <button 
              onClick={() => setEmailNotifs(!emailNotifs)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${emailNotifs ? 'bg-primary' : 'bg-secondary'}`}
            >
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${emailNotifs ? 'translate-x-6' : 'translate-x-1'}`} />
            </button>
          </div>
          
          <div className="w-full h-px bg-border"></div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-base font-medium flex items-center gap-2">
                Anonymous Data Sharing
                <Shield className="w-4 h-4 text-primary" />
              </label>
              <p className="text-sm text-foreground/60">Help us improve by sharing anonymous usage statistics.</p>
            </div>
            <button 
              onClick={() => setDataSharing(!dataSharing)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${dataSharing ? 'bg-primary' : 'bg-secondary'}`}
            >
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${dataSharing ? 'translate-x-6' : 'translate-x-1'}`} />
            </button>
          </div>
        </div>
      </div>

    </div>
  );
}
