import React, { useMemo, useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { useMemory } from '../contexts/MemoryContext';
import CharacterMemoryTransferPanel from './CharacterMemoryTransferPanel';
import MemoryCuratorPanel from './MemoryCuratorPanel';
import MemoryEditor from './MemoryEditor';
import PersonaRealignmentPanel from './PersonaRealignmentPanel';

const TOOL_CHOICES = [
  {
    id: 'refresh-character',
    label: 'Refresh a character',
    description: 'Update how a character responds to you from your shared history.',
  },
  {
    id: 'clean-profile',
    label: 'Clean my memories',
    description: 'Merge duplicates and remove noise from facts saved about you.',
  },
  {
    id: 'clean-character',
    label: 'Clean character memories',
    description: 'Review what one character has learned from its conversations with you.',
  },
  {
    id: 'move-character',
    label: 'Move character memories',
    description: 'Give one character a copy of another character’s continuity.',
  },
  {
    id: 'browse-profile',
    label: 'Browse saved memories',
    description: 'Inspect, correct or delete individual memories yourself.',
  },
];

export default function MemoryToolsHub() {
  const { MEMORY_API_URL, portsReady, storageHydrated, characters = [] } = useApp();
  const { userProfile, activeProfileId } = useMemory();
  const apiReady = portsReady && storageHydrated;
  const [activeTool, setActiveTool] = useState('refresh-character');
  const [memoryListRefreshKey, setMemoryListRefreshKey] = useState(0);
  const bumpMemories = () => setMemoryListRefreshKey((key) => key + 1);
  const selectedTool = useMemo(
    () => TOOL_CHOICES.find((tool) => tool.id === activeTool) || TOOL_CHOICES[0],
    [activeTool]
  );

  return (
    <div className="memory-tools-hub mx-auto max-w-6xl space-y-6 pb-16">
      <header className="rounded-xl border border-border bg-card/80 p-5 shadow-sm">
        <h1 className="text-2xl font-semibold tracking-tight">Memory tools</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
          Mirid keeps facts about you with your user profile, while each character keeps its own separate record of your shared history. Use these tools to review, correct or move that information. Nothing is replaced without your confirmation.
        </p>
      </header>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5" role="navigation" aria-label="Memory tools">
        {TOOL_CHOICES.map((tool) => {
          const selected = tool.id === activeTool;
          return (
            <button
              key={tool.id}
              type="button"
              aria-pressed={selected}
              onClick={() => setActiveTool(tool.id)}
              className={`rounded-xl border p-4 text-left transition-colors ${
                selected
                  ? 'border-primary bg-primary/10 shadow-sm'
                  : 'border-border bg-card/60 hover:border-primary/50 hover:bg-muted/30'
              }`}
            >
              <span className="block text-sm font-semibold">{tool.label}</span>
              <span className="mt-1 block text-xs leading-relaxed text-muted-foreground">{tool.description}</span>
            </button>
          );
        })}
      </div>

      <section aria-label={selectedTool.label}>
        {activeTool === 'refresh-character' ? <PersonaRealignmentPanel /> : null}

        {activeTool === 'clean-profile' ? (
          <div className="space-y-3">
            <div>
              <h2 className="text-lg font-semibold">Clean up what Mirid remembers about you</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                Your user-profile memories can follow you across characters. A character you choose reviews the list in a familiar voice, but the result must remain grounded in the saved facts.
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
          </div>
        ) : null}

        {activeTool === 'clean-character' ? (
          <div className="space-y-3">
            <div>
              <h2 className="text-lg font-semibold">Clean up what a character remembers</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                Select whose memory should be cleaned, then choose the character voice that should conduct the review. These choices can be different: one is the memory being edited; the other preserves in-world continuity while reviewing it.
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
          </div>
        ) : null}

        {activeTool === 'move-character' ? (
          <CharacterMemoryTransferPanel
            apiUrl={MEMORY_API_URL}
            apiReady={apiReady}
            activeProfileId={activeProfileId}
            characters={characters}
            onApplied={bumpMemories}
          />
        ) : null}

        {activeTool === 'browse-profile' ? (
          <div className="space-y-3">
            <div>
              <h2 className="text-lg font-semibold">Browse saved profile memories</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                Search the factual memories attached to your current user profile. Edit individual entries when an automated review would be unnecessary.
              </p>
            </div>
            <MemoryEditor hideCuratorPanels memoryListRefreshKey={memoryListRefreshKey} />
          </div>
        ) : null}
      </section>
    </div>
  );
}
