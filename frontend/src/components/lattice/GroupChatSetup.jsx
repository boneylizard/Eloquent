import React, { useState } from 'react';
import { X } from 'lucide-react';
import { usePool } from '../../contexts/PoolContext';

export default function GroupChatSetup({ onClose }) {
  const { poolCharacters, startGroupChat } = usePool();
  const eligible = poolCharacters.filter(c => c.dating_profile?.section_affinity?.length > 0);
  const [selectedIds, setSelectedIds] = useState([]);
  const [topic, setTopic] = useState('');

  const toggleChar = (id) => {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  const handleStart = () => {
    if (selectedIds.length < 2) return;
    startGroupChat(selectedIds, topic);
    onClose?.();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-card border rounded-2xl w-full max-w-md max-h-[85vh] overflow-y-auto shadow-2xl animate-in slide-in-from-bottom duration-300 p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-bold">Start Group Chat</h2>
          <button onClick={onClose} className="w-7 h-7 rounded-full hover:bg-muted flex items-center justify-center">
            <X className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs text-muted-foreground mb-4">Select 2-4 characters to chat with in a group.</p>

        <div className="space-y-1 max-h-60 overflow-y-auto mb-4">
          {eligible.map(char => {
            const isSelected = selectedIds.includes(char.id);
            return (
              <button
                key={char.id}
                onClick={() => toggleChar(char.id)}
                className={`w-full flex items-center gap-3 p-2.5 rounded-xl transition-all text-left ${
                  isSelected ? 'bg-primary/10 border border-primary/30' : 'bg-muted/30 hover:bg-muted/50 border border-transparent'
                }`}
              >
                <div className={`w-5 h-5 rounded border-2 flex items-center justify-center shrink-0 transition-colors ${
                  isSelected ? 'border-primary bg-primary' : 'border-muted-foreground/30'
                }`}>
                  {isSelected && <div className="w-2 h-2 rounded-sm bg-white" />}
                </div>
                <div className="w-8 h-8 rounded-full bg-muted overflow-hidden flex items-center justify-center text-xs font-bold shrink-0">
                  {char.avatar ? <img src={char.avatar} alt="" className="w-full h-full object-cover" /> : char.name?.[0] || '?'}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-semibold truncate">{char.name}</div>
                  <div className="text-[9px] text-muted-foreground truncate">
                    {(char.dating_profile?.section_affinity || []).join(', ')}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        <input
          value={topic}
          onChange={e => setTopic(e.target.value)}
          placeholder="Optional: set a topic (e.g., 'What makes you feel alive?')"
          className="w-full h-9 text-xs bg-muted border rounded-xl px-3 outline-none focus:border-primary/50 transition-colors mb-4"
        />

        <div className="flex gap-2">
          <button onClick={onClose} className="flex-1 py-2.5 rounded-xl bg-muted text-xs font-semibold hover:bg-muted/80 transition-colors">
            Cancel
          </button>
          <button
            onClick={handleStart}
            disabled={selectedIds.length < 2}
            className="flex-1 py-2.5 rounded-xl bg-primary text-primary-foreground text-xs font-semibold hover:bg-primary/90 transition-colors disabled:opacity-40"
          >
            Start Group Chat ({selectedIds.length})
          </button>
        </div>
      </div>
    </div>
  );
}
