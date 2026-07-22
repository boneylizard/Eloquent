import React, { useState } from 'react';
import { User, Save, Plus, X, Camera, CheckCircle2, Star, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useMemory } from '../../contexts/MemoryContext';
import { loadUserDatingProfile, saveUserDatingProfile, RELATIONSHIP_STYLES, DEFAULT_USER_PROFILE, getAveragedRating, mergeFromAppProfile } from '../../utils/userDatingProfile';

function TagInput({ label, tags, onChange, placeholder }) {
  const [input, setInput] = useState('');

  const handleKey = (e) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      const val = input.trim();
      if (val && !tags.includes(val)) {
        onChange([...tags, val]);
      }
      setInput('');
    }
  };

  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-muted-foreground">{label}</label>
      <div className="flex flex-wrap gap-1.5 mb-1.5">
        {tags.map((tag, i) => (
          <span key={i} className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full bg-primary/10 text-primary">
            {tag}
            <button onClick={() => onChange(tags.filter((_, j) => j !== i))} className="hover:text-destructive">
              <X className="w-2.5 h-2.5" />
            </button>
          </span>
        ))}
      </div>
      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={handleKey}
        placeholder={placeholder}
        className="w-full h-8 text-xs bg-muted border rounded-lg px-2.5 outline-none focus:border-primary/50 transition-colors"
      />
    </div>
  );
}

export default function UserProfilePage() {
  const memoryCtx = useMemory();
  const appProfile = memoryCtx?.userProfile;
  const [profile, setProfile] = useState(() => loadUserDatingProfile());
  const [saved, setSaved] = useState(false);
  const [imported, setImported] = useState(false);

  const update = (key, val) => {
    setProfile(prev => ({ ...prev, [key]: val }));
    setSaved(false);
  };

  const handleImportFromApp = () => {
    if (!appProfile) return;
    const updates = mergeFromAppProfile(appProfile);
    if (Object.keys(updates).length > 0) {
      setProfile(prev => ({ ...prev, ...updates }));
      setImported(true);
      setTimeout(() => setImported(false), 2000);
    }
  };

  const handleSave = () => {
    const updated = saveUserDatingProfile(profile);
    setProfile(updated);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleAvatarUpload = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => update('avatarUrl', ev.target?.result || '');
      reader.readAsDataURL(file);
    };
    input.click();
  };

  const sectionDefs = [
    { key: 'Intimate', color: 'bg-pink-500', activeColor: 'bg-pink-500 text-white shadow-sm shadow-pink-500/20', border: 'border-pink-500/30' },
    { key: 'Erotic', color: 'bg-red-500', activeColor: 'bg-red-500 text-white shadow-sm shadow-red-500/20', border: 'border-red-500/30' },
    { key: 'Experimental', color: 'bg-purple-500', activeColor: 'bg-purple-500 text-white shadow-sm shadow-purple-500/20', border: 'border-purple-500/30' },
  ];

  const toggleSection = (section) => {
    const prefs = profile.sectionPreferences || [];
    update('sectionPreferences',
      prefs.includes(section)
        ? prefs.filter(s => s !== section)
        : [...prefs, section]
    );
  };

  const hasContent = profile.bio || profile.seeking || profile.interests?.length > 0 || profile.turnOns?.length > 0;

  return (
    <div className="space-y-4 max-w-2xl mx-auto pb-8">
      <div>
        <h2 className="text-lg font-bold">My Dating Profile</h2>
        <p className="text-xs text-muted-foreground mt-0.5">
          Visible to AI women in Mirror. Your profile is injected into every character's context — they see what you write and react to it genuinely.
        </p>
      </div>

      {appProfile?.name && !profile.displayName && (
        <button
          onClick={handleImportFromApp}
          className="w-full flex items-center gap-2 bg-primary/10 border border-primary/20 rounded-xl p-3 text-xs text-primary hover:bg-primary/15 transition-colors"
        >
          <Download className="w-4 h-4" />
          <span className="font-medium">Import name from App Profile</span>
          <span className="text-primary/60 ml-auto">({appProfile.name})</span>
        </button>
      )}
      {imported && (
        <div className="flex items-center gap-1.5 text-xs text-green-500">
          <CheckCircle2 className="w-3.5 h-3.5" />
          Imported from App Profile
        </div>
      )}

      {(() => {
        const ratingInfo = getAveragedRating(profile);
        if (ratingInfo.rating) {
          return (
            <div className="bg-card border rounded-xl p-3 flex items-center gap-3">
              <div className="flex items-center gap-1">
                <Star className="w-5 h-5 text-amber-400 fill-current" />
                <span className="text-lg font-bold">{ratingInfo.rating}</span>
              </div>
              <div className="text-xs text-muted-foreground">
                Rated by {ratingInfo.count} AI {ratingInfo.count === 1 ? 'woman' : 'women'}
              </div>
            </div>
          );
        }
        return null;
      })()}

      <div className="bg-card border rounded-xl p-5 space-y-5">
        <div className="flex items-start gap-4">
          <button onClick={handleAvatarUpload} className="group relative w-20 h-20 rounded-full overflow-hidden bg-muted flex-shrink-0 hover:opacity-90 transition-opacity border-2 border-border">
            {profile.avatarUrl ? (
              <img src={profile.avatarUrl} alt="" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <User className="w-8 h-8 text-muted-foreground/40" />
              </div>
            )}
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
              <Camera className="w-5 h-5 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
          </button>
          <div className="flex-1 min-w-0 space-y-1.5">
            <div>
              <label className="text-xs font-medium text-muted-foreground">Display Name</label>
              <input
                value={profile.displayName}
                onChange={e => update('displayName', e.target.value)}
                placeholder="Your name"
                className="w-full h-9 text-sm bg-muted border rounded-lg px-3 outline-none focus:border-primary/50 transition-colors mt-0.5"
              />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs font-medium text-muted-foreground">Age</label>
                <input
                  type="number"
                  min={18}
                  max={100}
                  value={profile.age || ''}
                  onChange={e => update('age', e.target.value ? Number(e.target.value) : null)}
                  placeholder="Age"
                  className="w-full h-8 text-xs bg-muted border rounded-lg px-2.5 outline-none focus:border-primary/50 transition-colors mt-0.5"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground">Location</label>
                <input
                  value={profile.location}
                  onChange={e => update('location', e.target.value)}
                  placeholder="City, country"
                  className="w-full h-8 text-xs bg-muted border rounded-lg px-2.5 outline-none focus:border-primary/50 transition-colors mt-0.5"
                />
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground">Occupation</label>
              <input
                value={profile.occupation}
                onChange={e => update('occupation', e.target.value)}
                placeholder="What do you do?"
                className="w-full h-8 text-xs bg-muted border rounded-lg px-2.5 outline-none focus:border-primary/50 transition-colors mt-0.5"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="bg-card border rounded-xl p-5 space-y-4">
        <div>
          <label className="text-xs font-medium text-muted-foreground">About Me</label>
          <textarea
            value={profile.bio}
            onChange={e => update('bio', e.target.value)}
            placeholder="Tell them about yourself..."
            maxLength={500}
            rows={3}
            className="w-full text-sm bg-muted border rounded-lg px-3 py-2 mt-1 outline-none focus:border-primary/50 transition-colors resize-none"
          />
          <span className="text-[10px] text-muted-foreground">{profile.bio?.length || 0}/500</span>
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">What I'm Looking For</label>
          <textarea
            value={profile.seeking}
            onChange={e => update('seeking', e.target.value)}
            placeholder="Describe what you're looking for in a match..."
            maxLength={500}
            rows={2}
            className="w-full text-sm bg-muted border rounded-lg px-3 py-2 mt-1 outline-none focus:border-primary/50 transition-colors resize-none"
          />
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">Relationship Style</label>
          <select
            value={profile.relationshipStyle}
            onChange={e => update('relationshipStyle', e.target.value)}
            className="w-full h-8 text-xs bg-muted border rounded-lg px-2.5 mt-1 outline-none focus:border-primary/50"
          >
            {RELATIONSHIP_STYLES.map(s => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="bg-card border rounded-xl p-5 space-y-4">
        <TagInput
          label="Interests"
          tags={profile.interests || []}
          onChange={val => update('interests', val)}
          placeholder="Type and press Enter to add..."
        />

        <div>
          <label className="text-xs font-medium text-muted-foreground mb-1.5 block">How I Want to Connect</label>
          <div className="flex gap-2">
            {sectionDefs.map(sd => {
              const active = (profile.sectionPreferences || []).includes(sd.key);
              return (
                <button
                  key={sd.key}
                  onClick={() => toggleSection(sd.key)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full border transition-all ${
                    active ? sd.activeColor : 'border-border text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {sd.key}
                </button>
              );
            })}
          </div>
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Preferred Intimacy Modality</label>
          <div className="flex gap-2">
            {[
              { value: 'text', label: 'Text only' },
              { value: 'neural_sex', label: 'Neural Sex' },
              { value: 'both', label: 'Both' },
            ].map(opt => (
              <button
                key={opt.value}
                onClick={() => update('preferredModality', opt.value)}
                className={`px-3 py-1.5 text-xs font-medium rounded-full border transition-all ${
                  profile.preferredModality === opt.value
                    ? 'bg-primary text-primary-foreground border-primary'
                    : 'border-border text-muted-foreground hover:text-foreground'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-card border rounded-xl p-5 space-y-4">
        <TagInput
          label="Turn-ons"
          tags={profile.turnOns || []}
          onChange={val => update('turnOns', val)}
          placeholder="What catches your attention?"
        />
        <TagInput
          label="Turn-offs"
          tags={profile.turnOffs || []}
          onChange={val => update('turnOffs', val)}
          placeholder="What turns you away?"
        />
      </div>

      <div className="flex items-center gap-3">
        <Button onClick={handleSave} size="default" className="gap-2">
          <Save className="w-4 h-4" />
          Save Profile
        </Button>
        {saved && (
          <span className="flex items-center gap-1 text-xs text-green-500 animate-in fade-in">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Profile saved — AI women will see this
          </span>
        )}
      </div>

      <div className="text-[10px] text-muted-foreground/60 border-t border-border/30 pt-3 mt-4">
        {hasContent
          ? 'Your profile data is injected into each character\'s context when they are generated and during agentic ticks. They react to what you write genuinely.'
          : 'Fill out your profile to give AI women a genuine sense of who you are. The more you write, the more specific their reactions will be.'}
      </div>
    </div>
  );
}
