import React, { useEffect, useState, useCallback } from 'react';
import { Users, Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { usePool } from '../../contexts/PoolContext';

const BASE_DUMMIES = [
  { name: 'Marcus', initial: 'M', desc: 'Confident, direct', style: 'bg-blue-500/10 text-blue-400', id: 'dummy_marcus', avatar_url: '/static/dummy_avatars/dummy_marcus.svg' },
  { name: 'Liam', initial: 'L', desc: 'Thoughtful, intense', style: 'bg-emerald-500/10 text-emerald-400', id: 'dummy_liam', avatar_url: '/static/dummy_avatars/dummy_liam.svg' },
  { name: 'Rafe', initial: 'R', desc: 'Charismatic, playful', style: 'bg-amber-500/10 text-amber-400', id: 'dummy_rafe', avatar_url: '/static/dummy_avatars/dummy_rafe.svg' },
  { name: 'Khalid', initial: 'K', desc: 'Grounded, observant', style: 'bg-rose-500/10 text-rose-400', id: 'dummy_khalid', avatar_url: '/static/dummy_avatars/dummy_khalid.svg' },
  { name: 'Ethan', initial: 'E', desc: 'Eager, intense', style: 'bg-cyan-500/10 text-cyan-400', id: 'dummy_ethan', avatar_url: '/static/dummy_avatars/dummy_ethan.svg' },
];

export default function DummyRivalsPanel() {
  const { dummyRealism, setDummyRealism, dummyAgency, setDummyAgency, generatedDummies, generateDummyRival } = usePool();

  const allDummies = [...BASE_DUMMIES, ...(generatedDummies || []).map(d => ({
    name: d.name,
    initial: d.name?.[0] || '?',
    desc: d.personality?.split('.')[0] || 'New rival',
    style: 'bg-gray-500/10 text-gray-400 border-dashed',
    id: d.id || d.name,
    avatar_url: d.avatar_url || '/static/dummy_avatars/dummy_marcus.svg',
    generated: true,
  }))];

  return (
    <div className="space-y-4">
      <div className="bg-card border rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-blue-500" />
            <h3 className="text-base font-bold">Dummy Rivals</h3>
          </div>
          <Button size="sm" variant="outline" onClick={generateDummyRival} className="gap-1">
            <RefreshCw className="w-3 h-3" />
            Generate New
          </Button>
        </div>

        <p className="text-xs text-muted-foreground leading-relaxed">
          Simulated male profiles that create social texture and competition in the pool.
          Female AIs perceive them as real users — they do not know which profiles are simulated.
          New dummies can be generated to keep the pool dynamic.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <label className="font-medium">Realism Level</label>
              <span className="font-mono text-muted-foreground">{dummyRealism}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={dummyRealism}
              onChange={e => setDummyRealism(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-full appearance-none cursor-pointer accent-primary"
            />
            <p className="text-[10px] text-muted-foreground/60">
              {dummyRealism < 30 ? 'Basic, predictable responses' :
               dummyRealism < 60 ? 'Moderately convincing social behavior' :
               'Highly nuanced, near-realistic interactions'}
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <label className="font-medium">Agency Level</label>
              <span className="font-mono text-muted-foreground">{dummyAgency}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={dummyAgency}
              onChange={e => setDummyAgency(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-full appearance-none cursor-pointer accent-primary"
            />
            <p className="text-[10px] text-muted-foreground/60">
              {dummyAgency < 30 ? 'Passive — respond only when spoken to' :
               dummyAgency < 60 ? 'Occasionally initiate conversations' :
               'Proactive — message and compete autonomously'}
            </p>
          </div>
        </div>
      </div>

      <div className="bg-card border rounded-xl p-4">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          Active Dummy Profiles ({allDummies.length})
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
          {allDummies.map(d => (
            <div key={d.id || d.name} className={`flex flex-col items-center p-3 rounded-lg border ${d.generated ? 'border-dashed border-muted-foreground/20' : 'bg-muted/50 border-border/30'}`}>
              <div className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold mb-1.5 overflow-hidden">
                <img
                  src={d.avatar_url}
                  alt={d.name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.parentElement.className = d.style + ' w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold mb-1.5';
                    e.target.parentElement.textContent = d.initial;
                  }}
                />
              </div>
              <span className="text-xs font-medium">{d.name}</span>
              <span className="text-[9px] text-muted-foreground text-center mt-0.5">{d.desc}</span>
              {d.generated && <span className="text-[8px] text-muted-foreground/50 mt-1">Auto-generated</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
