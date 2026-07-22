import React, { useState, useEffect, useCallback } from 'react';
import { Brain, Save, Trash2, Plus, RotateCcw, CheckCircle2, AlertCircle, Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import { useAgenticProfile } from '../contexts/AgenticProfileContext';
import { getBackendUrl } from '../config/api';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';

const SettingsSection = ({ title, description, children, actions }) => (
  <div className="rounded-2xl border border-border/70 bg-card/60 shadow-sm">
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2 border-b border-border/60 px-5 py-4">
      <div>
        <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">{title}</p>
        {description ? (
          <p className="text-sm text-foreground/80 mt-1">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
    <div className="p-5 space-y-3">{children}</div>
  </div>
);

const LABEL_GROUPS = [
  {
    title: 'Event / JSON keys (internal routing)',
    keys: ['LABEL_TACTILE_OUTREACH', 'LABEL_CHARACTER_SIGNAL'],
  },
  {
    title: 'Glass UI Display Headings (user-facing)',
    keys: ['LABEL_TACTILE_DISPLAY', 'LABEL_SIGNAL_DISPLAY'],
  },
  {
    title: 'Sub-field keys (inside JSON objects)',
    keys: ['LABEL_POSE', 'LABEL_GESTURE', 'LABEL_PROXIMITY', 'LABEL_COVERT_ACTION', 'LABEL_VOICE_THIS_TURN'],
  },
  {
    title: 'Dashboard chip labels',
    keys: ['dashboard_lubrication', 'dashboard_pupils', 'dashboard_position', 'dashboard_breath', 'dashboard_tension'],
  },
  {
    title: 'Gauge labels',
    keys: ['heat_gauge', 'dominance_gauge', 'trap_gauge', 'posture_label'],
  },
];

const PROMPT_KEYS = [
  { key: 'contextual_analysis', label: 'Contextual Analysis (Step 1)', height: '300px' },
  { key: 'somatic_generation', label: 'Somatic Generation (Step 2)', height: '300px' },
  { key: 'directive_block_template', label: 'Directive Block (Step 3 template)', height: '200px' },
  { key: 'planning_generation', label: 'Planning Generation (Step 3b)', height: '200px' },
];

export default function AgenticProfileSettings() {
  const {
    profileId, profile, profiles, loaded, loading,
    labels, displayConfig,
    fetchProfiles, loadProfile, saveProfile, deleteProfile,
  } = useAgenticProfile();

  const [editName, setEditName] = useState('');
  const [editDesc, setEditDesc] = useState('');
  const [editLabels, setEditLabels] = useState({});
  const [editPrompts, setEditPrompts] = useState({});
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [testing, setTesting] = useState(false);
  const [newProfileName, setNewProfileName] = useState('');
  const [error, setError] = useState('');
  const [expandedSections, setExpandedSections] = useState({
    labels: true,
    prompts: false,
    test: false,
  });

  useEffect(() => {
    if (profile) {
      setEditName(profile.name || '');
      setEditDesc(profile.description || '');
      setEditLabels({ ...(profile.labels || {}) });
      setEditPrompts({ ...(profile.prompts || {}) });
      setDirty(false);
      setTestResult(null);
    }
  }, [profile]);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const handleLabelChange = (key, value) => {
    setEditLabels(prev => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  const handlePromptChange = (key, value) => {
    setEditPrompts(prev => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  const handleSave = async () => {
    if (profileId === '_default') {
      setError('Cannot modify the default profile. Create a new profile first.');
      return;
    }
    setSaving(true);
    setError('');
    const data = {
      name: editName,
      description: editDesc,
      labels: editLabels,
      prompts: editPrompts,
    };
    const ok = await saveProfile(profileId, data);
    setSaving(false);
    if (ok) {
      setDirty(false);
      await loadProfile(profileId);
    } else {
      setError('Failed to save profile');
    }
  };

  const handleCreate = async () => {
    const name = newProfileName.trim();
    if (!name) return;
    const id = name.toLowerCase().replace(/[^a-z0-9_-]/g, '_');
    const data = {
      id,
      name,
      description: '',
      labels: { ...labels },
      prompts: {},
    };
    const ok = await saveProfile(id, data);
    if (ok) {
      setNewProfileName('');
      await loadProfile(id);
    }
  };

  const handleDelete = async () => {
    if (profileId === '_default') return;
    if (!window.confirm(`Delete profile "${editName}"? This cannot be undone.`)) return;
    await deleteProfile(profileId);
  };

  const handleResetToDefaults = async () => {
    if (profileId === '_default') return;
    await loadProfile('_default');
    setEditLabels({ ...(profile?.labels || {}) });
    setEditPrompts({});
    setDirty(true);
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const base = getBackendUrl();
      const res = await fetch(`${base}/agentic/profiles/${encodeURIComponent(profileId)}`, {
        method: 'GET',
      });
      if (res.ok) {
        const data = await res.json();
        setTestResult({ ok: true, profile: data.profile });
      } else {
        setTestResult({ ok: false, error: `HTTP ${res.status}` });
      }
    } catch (err) {
      setTestResult({ ok: false, error: err.message });
    }
    setTesting(false);
  };

  const isDefault = profileId === '_default';

  return (
    <div className="space-y-6 px-0">
      <div className="flex items-center gap-2 mb-4">
        <Brain size={18} className="text-primary" />
        <span className="text-sm font-semibold">Agentic Profiles</span>
      </div>

      {error && (
        <div className="p-3 text-sm rounded-lg bg-destructive/10 text-destructive border border-destructive/20">
          {error}
        </div>
      )}

      {/* Profile Selector */}
      <SettingsSection
        title="Profile Selection"
        description="Choose which agentic profile to edit"
        actions={
          isDefault ? null : (
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={handleResetToDefaults}>
                <RotateCcw size={12} className="mr-1" />
                Reset to Defaults
              </Button>
              <Button variant="destructive" size="sm" onClick={handleDelete}>
                <Trash2 size={12} className="mr-1" />
                Delete
              </Button>
            </div>
          )
        }
      >
        <Select value={profileId} onValueChange={(v) => loadProfile(v)}>
          <SelectTrigger className="w-full max-w-md">
            <SelectValue placeholder="Select a profile..." />
          </SelectTrigger>
          <SelectContent>
            {profiles.map((p) => (
              <SelectItem key={p.id} value={p.id}>
                {p.name} {p.id === '_default' ? '(default)' : ''}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {!isDefault && (
          <div className="flex items-center gap-2 mt-3">
            <Input
              placeholder="Profile name"
              value={editName}
              onChange={(e) => { setEditName(e.target.value); setDirty(true); }}
              className="max-w-xs"
            />
            <Input
              placeholder="Description (optional)"
              value={editDesc}
              onChange={(e) => { setEditDesc(e.target.value); setDirty(true); }}
              className="max-w-sm"
            />
          </div>
        )}
      </SettingsSection>

      {loading && (
        <div className="flex items-center justify-center py-8">
          <Loader2 size={20} className="animate-spin text-primary" />
          <span className="ml-2 text-sm text-muted-foreground">Loading profile...</span>
        </div>
      )}

      {!loaded && !loading && (
        <div className="text-sm text-muted-foreground py-4">
          Could not load profiles. Is the backend running?
        </div>
      )}

      {profile && (
        <>
          {/* Labels Section */}
          <div className="rounded-2xl border border-border/70 bg-card/60 shadow-sm">
            <button
              onClick={() => toggleSection('labels')}
              className="flex items-center justify-between w-full px-5 py-4 border-b border-border/60"
            >
              <div className="text-left">
                <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">Labels & Display Names</p>
                <p className="text-sm text-foreground/80 mt-1">Customize all UI-facing label strings</p>
              </div>
              {expandedSections.labels ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </button>
            {expandedSections.labels && (
              <div className="p-5 space-y-4">
                {LABEL_GROUPS.map((group) => (
                  <div key={group.title} className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">{group.title}</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {group.keys.map((key) => (
                        <div key={key} className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground min-w-[140px] font-mono">{key}</span>
                          <Input
                            value={editLabels[key] || ''}
                            onChange={(e) => handleLabelChange(key, e.target.value)}
                            className="font-mono text-xs"
                            placeholder={key}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Prompts Section */}
          <div className="rounded-2xl border border-border/70 bg-card/60 shadow-sm">
            <button
              onClick={() => toggleSection('prompts')}
              className="flex items-center justify-between w-full px-5 py-4 border-b border-border/60"
            >
              <div className="text-left">
                <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">Prompt Templates</p>
                <p className="text-sm text-foreground/80 mt-1">Customize the LLM prompts for each pipeline step. Leave blank to use defaults.</p>
              </div>
              {expandedSections.prompts ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </button>
            {expandedSections.prompts && (
              <div className="p-5 space-y-4">
                <div className="p-3 rounded-lg bg-muted/30 text-xs text-muted-foreground border border-border/40">
                  Available template variables: {'{posture}'}, {'{dominance}'}, {'{heat}'}, {'{trap}'}, {'{emotional_state}'}, {'{physical_state}'}, {'{trajectory}'}, {'{full_context}'}, {'{shadow_state}'}, {'{history}'}, {'{user_text}'}, {'{character}'}, {'{dashboard}'}, {'{internal_state}'}, {'{external_state}'}, {'{somatic_narrative}'}, {'{behavioral_cues}'}, {'{LABEL_TACTILE_OUTREACH}'}, {'{LABEL_CHARACTER_SIGNAL}'}, {'{LABEL_POSE}'}, {'{LABEL_GESTURE}'}, {'{LABEL_PROXIMITY}'}, {'{LABEL_COVERT_ACTION}'}, {'{LABEL_VOICE_THIS_TURN}'}, {'{LABEL_TACTILE_DISPLAY}'}, {'{LABEL_SIGNAL_DISPLAY}'}
                </div>
                {PROMPT_KEYS.map((pk) => (
                  <div key={pk.key}>
                    <p className="text-xs font-medium text-muted-foreground mb-1">{pk.label}</p>
                    <Textarea
                      value={editPrompts[pk.key] || ''}
                      onChange={(e) => handlePromptChange(pk.key, e.target.value)}
                      className="font-mono text-xs"
                      style={{ minHeight: pk.height }}
                      placeholder={`Leave empty to use the built-in default ${pk.label}`}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Save Button */}
          {dirty && !isDefault && (
            <div className="flex items-center gap-3">
              <Button onClick={handleSave} disabled={saving}>
                {saving ? <Loader2 size={14} className="animate-spin mr-1" /> : <Save size={14} className="mr-1" />}
                Save Profile
              </Button>
              <span className="text-xs text-muted-foreground">Unsaved changes</span>
            </div>
          )}

          {isDefault && (
            <div className="p-4 rounded-lg bg-muted/20 text-sm text-muted-foreground border border-border/40">
              The default profile cannot be edited. Create a new profile to customize.
            </div>
          )}

          {/* Test Section */}
          <div className="rounded-2xl border border-border/70 bg-card/60 shadow-sm">
            <button
              onClick={() => toggleSection('test')}
              className="flex items-center justify-between w-full px-5 py-4 border-b border-border/60"
            >
              <div className="text-left">
                <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">Test Profile</p>
                <p className="text-sm text-foreground/80 mt-1">Verify the profile is loaded correctly from the backend</p>
              </div>
              {expandedSections.test ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </button>
            {expandedSections.test && (
              <div className="p-5 space-y-3">
                <Button variant="outline" size="sm" onClick={handleTest} disabled={testing}>
                  {testing ? <Loader2 size={12} className="animate-spin mr-1" /> : <Brain size={12} className="mr-1" />}
                  Test Profile
                </Button>
                {testResult && (
                  <div className={`p-3 rounded-lg text-xs font-mono border ${
                    testResult.ok
                      ? 'bg-green-500/5 border-green-500/20 text-green-600 dark:text-green-400'
                      : 'bg-destructive/5 border-destructive/20 text-destructive'
                  }`}>
                    {testResult.ok ? (
                      <div>
                        <div className="flex items-center gap-1 mb-1">
                          <CheckCircle2 size={12} />
                          <span className="font-semibold">Profile loaded successfully</span>
                        </div>
                        <div className="opacity-70">
                          Name: {testResult.profile?.name}<br />
                          Labels: {Object.keys(testResult.profile?.labels || {}).length} entries<br />
                          Custom prompts: {Object.entries(testResult.profile?.prompts || {}).filter(([, v]) => v).length} defined
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center gap-1">
                        <AlertCircle size={12} />
                        <span>Failed: {testResult.error}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}

      {/* Create New Profile */}
      <SettingsSection title="Create New Profile" description="Create a new profile based on the current defaults">
        <div className="flex items-center gap-2">
          <Input
            placeholder="New profile name..."
            value={newProfileName}
            onChange={(e) => setNewProfileName(e.target.value)}
            className="max-w-xs"
            onKeyDown={(e) => { if (e.key === 'Enter') handleCreate(); }}
          />
          <Button variant="outline" size="sm" onClick={handleCreate} disabled={!newProfileName.trim()}>
            <Plus size={12} className="mr-1" />
            Create
          </Button>
        </div>
      </SettingsSection>
    </div>
  );
}
