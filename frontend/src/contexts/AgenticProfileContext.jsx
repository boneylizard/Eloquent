import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { getBackendUrl } from '../config/api';

const AgenticProfileContext = createContext(null);
export const useAgenticProfile = () => useContext(AgenticProfileContext);

const DEFAULT_LABELS = {
  LABEL_TACTILE_OUTREACH: 'tactile_outreach',
  LABEL_CHARACTER_SIGNAL: 'character_signal',
  LABEL_TACTILE_DISPLAY: 'Tactile Outreach',
  LABEL_SIGNAL_DISPLAY: 'Character Signal',
  LABEL_POSE: 'pose',
  LABEL_GESTURE: 'gesture_narrative',
  LABEL_PROXIMITY: 'proximity',
  LABEL_COVERT_ACTION: 'covert_action',
  LABEL_VOICE_THIS_TURN: 'voice_this_turn',
};

const DEFAULT_GAUGE_CONFIG = {
  heat: { label: 'Heat Index', color: 'rgba(255, 130, 100, 0.8)' },
  dominance: { label: 'Dominance', color: 'rgba(120, 180, 255, 0.8)' },
  trap: { label: 'Trap Progress', color: 'rgba(255, 120, 200, 0.8)' },
};

const DEFAULT_DASHBOARD_CHIPS = {
  lubrication_level: { label: 'Lubrication', color: 'rgba(255, 120, 200, 0.8)' },
  pupil_dilation: { label: 'Pupils', color: 'rgba(120, 200, 255, 0.8)' },
  spatial_position: { label: 'Position', color: 'rgba(100, 220, 150, 0.8)' },
  breath_rate: { label: 'Breath', color: 'rgba(255, 220, 120, 0.8)' },
  muscle_tension: { label: 'Tension', color: 'rgba(255, 130, 100, 0.8)' },
};

const DEFAULT_DISPLAY_CONFIG = {
  gauges: DEFAULT_GAUGE_CONFIG,
  dashboard_chips: DEFAULT_DASHBOARD_CHIPS,
};

export function AgenticProfileProvider({ children }) {
  const [profileId, setProfileId] = useState('_default');
  const [profile, setProfile] = useState(null);
  const [profiles, setProfiles] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [loading, setLoading] = useState(false);

  const fetchProfiles = useCallback(async () => {
    try {
      const base = getBackendUrl();
      const res = await fetch(`${base}/agentic/profiles`);
      if (res.ok) {
        const data = await res.json();
        setProfiles(data.profiles || []);
      }
    } catch (err) {
      console.warn('[AgenticProfile] Failed to fetch profiles:', err);
    }
  }, []);

  const fetchProfile = useCallback(async (id) => {
    setLoading(true);
    try {
      const base = getBackendUrl();
      const res = await fetch(`${base}/agentic/profiles/${encodeURIComponent(id)}`);
      if (res.ok) {
        const data = await res.json();
        setProfile(data.profile || null);
        return data.profile || null;
      }
    } catch (err) {
      console.warn('[AgenticProfile] Failed to fetch profile:', err);
    } finally {
      setLoading(false);
      setLoaded(true);
    }
    return null;
  }, []);

  const loadProfile = useCallback(async (id) => {
    setProfileId(id || '_default');
    await fetchProfile(id || '_default');
  }, [fetchProfile]);

  const saveProfile = useCallback(async (id, data) => {
    try {
      const base = getBackendUrl();
      const res = await fetch(`${base}/agentic/profiles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, ...data }),
      });
      if (res.ok) {
        await fetchProfiles();
        return true;
      }
      const errText = await res.text().catch(() => res.statusText);
      console.warn('[AgenticProfile] Failed to save profile:', res.status, errText);
      return false;
    } catch (err) {
      console.warn('[AgenticProfile] Save error:', err);
      return false;
    }
  }, [fetchProfiles]);

  const deleteProfile = useCallback(async (id) => {
    try {
      const base = getBackendUrl();
      const res = await fetch(`${base}/agentic/profiles/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        await fetchProfiles();
        if (profileId === id) {
          await loadProfile('_default');
        }
        return true;
      }
      return false;
    } catch (err) {
      console.warn('[AgenticProfile] Delete error:', err);
      return false;
    }
  }, [fetchProfiles, loadProfile, profileId]);

  useEffect(() => {
    fetchProfiles();
    fetchProfile('_default');
  }, []);

  const labels = profile?.labels || DEFAULT_LABELS;
  const displayConfig = profile?.display_config || DEFAULT_DISPLAY_CONFIG;

  const value = {
    profileId,
    profile,
    profiles,
    loaded,
    loading,
    labels,
    displayConfig,
    fetchProfiles,
    fetchProfile,
    loadProfile,
    saveProfile,
    deleteProfile,
    setProfileId,
  };

  return (
    <AgenticProfileContext.Provider value={value}>
      {children}
    </AgenticProfileContext.Provider>
  );
}
