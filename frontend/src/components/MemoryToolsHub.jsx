import React, { useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { useMemory } from '../contexts/MemoryContext';
import PersonaRealignmentPanel from './PersonaRealignmentPanel';
import MemoryCuratorPanel from './MemoryCuratorPanel';
import MemoryEditor from './MemoryEditor';

/**
 * Single entry point for persona realignment, profile/agentic curators,
 * and the classic memory list (sidebar: Memory tools).
 */
export default function MemoryToolsHub() {
  const { MEMORY_API_URL, portsReady, storageHydrated, characters = [] } = useApp();
  const { userProfile, activeProfileId } = useMemory();
  const apiReady = portsReady && storageHydrated;
  const [memoryListRefreshKey, setMemoryListRefreshKey] = useState(0);
  const bumpMemories = () => setMemoryListRefreshKey((k) => k + 1);

  return (
    <div className="memory-tools-hub space-y-10 max-w-6xl mx-auto pb-16">
      <header className="rounded-xl border border-border bg-card/80 p-5 shadow-sm">
        <h1 className="text-2xl font-semibold tracking-tight">Memory tools</h1>
        <p className="text-muted-foreground text-sm leading-relaxed mt-2">
          Persona realignment (prompt builder), then profile and character curators, then browse memories.
          Open from the sidebar (Memory tools). The same realignment panel is under Settings → Persona realignment.
        </p>
        <nav className="flex flex-wrap gap-x-3 gap-y-1 items-center text-sm mt-4 text-primary">
          <a href="#persona-realignment" className="underline-offset-4 hover:underline">
            Persona realignment
          </a>
          <span className="text-muted-foreground" aria-hidden>
            ·
          </span>
          <a href="#profile-curator" className="underline-offset-4 hover:underline">
            Profile curator
          </a>
          <span className="text-muted-foreground" aria-hidden>
            ·
          </span>
          <a href="#agentic-curator" className="underline-offset-4 hover:underline">
            Character curator
          </a>
          <span className="text-muted-foreground" aria-hidden>
            ·
          </span>
          <a href="#memory-list" className="underline-offset-4 hover:underline">
            Memory list
          </a>
        </nav>
      </header>

      <section id="persona-realignment" className="scroll-mt-24 space-y-3">
        <div>
          <h2 className="text-lg font-semibold">1 · Persona realignment</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Sticky row on the panel: <strong>Build</strong>, then <strong>Run model</strong>, then <strong>Parse</strong>. Destructive save only inside the panel if you open it.
          </p>
        </div>
        <PersonaRealignmentPanel />
      </section>

      <section id="profile-curator" className="scroll-mt-24 space-y-3">
        <div>
          <h2 className="text-lg font-semibold">2 · Profile memory curator</h2>
          <p className="text-sm text-muted-foreground mt-1">
            In-character audit of memories tied to your user profile; apply rebuilds when satisfied.
          </p>
        </div>
        <MemoryCuratorPanel
          apiUrl={MEMORY_API_URL}
          apiReady={apiReady}
          activeProfileId={activeProfileId}
          userProfile={userProfile}
          characters={characters}
          scope="profile"
          onApplied={bumpMemories}
        />
      </section>

      <section id="agentic-curator" className="scroll-mt-24 space-y-3">
        <div>
          <h2 className="text-lg font-semibold">3 · Character (agentic) memory curator</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Same flow for NPC / assistant agentic memory: pick curator voice and target character, then build and apply.
          </p>
        </div>
        <MemoryCuratorPanel
          apiUrl={MEMORY_API_URL}
          apiReady={apiReady}
          activeProfileId={activeProfileId}
          userProfile={userProfile}
          characters={characters}
          scope="agentic"
          onApplied={bumpMemories}
        />
      </section>

      <section id="memory-list" className="scroll-mt-24 space-y-3">
        <div>
          <h2 className="text-lg font-semibold">4 · Browse &amp; edit memories</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Search, add, edit, dedupe. Profile settings live under User Profile in the manager below.
          </p>
        </div>
        <MemoryEditor hideCuratorPanels memoryListRefreshKey={memoryListRefreshKey} />
      </section>
    </div>
  );
}
