import React, { useEffect } from 'react';
import { MessageSquare, Trash2 } from 'lucide-react';
import { usePool } from '../../contexts/PoolContext';
import DMThreadView from './DMThreadView';

function formatTime(iso) {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 60000) return 'now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`;
  return new Date(iso).toLocaleDateString();
}

export default function DMThreads() {
  const { dmThreads, fetchDMThreads, activeDMThread, selectDMThread, closeDMThread, deleteDMThread, deleteAllDMThreads } = usePool();

  useEffect(() => { fetchDMThreads(); }, [fetchDMThreads]);

  if (activeDMThread) {
    return <DMThreadView thread={activeDMThread} onClose={closeDMThread} />;
  }

  return (
    <div className="space-y-3 max-w-2xl mx-auto pb-8">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-primary" />
            Direct Messages
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Persistent conversations with characters. Messages stay here, no time limit.
          </p>
        </div>
        {dmThreads.length > 0 && (
          <button
            onClick={() => {
              if (window.confirm('Delete all DM threads? This cannot be undone.')) deleteAllDMThreads();
            }}
            className="shrink-0 inline-flex items-center gap-1 rounded-full border border-red-500/20 px-2.5 py-1 text-[10px] text-red-400 hover:bg-red-500/10 transition-colors"
            title="Delete all DM threads"
          >
            <Trash2 className="w-3 h-3" />
            Delete all
          </button>
        )}
      </div>

      {dmThreads.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <img src="/logos/mirrorlogosamle2.webp" alt="" className="w-16 h-16 object-contain mb-3 opacity-30" />
          <p className="text-sm text-muted-foreground">No messages yet.</p>
          <p className="text-xs text-muted-foreground/60 mt-1">
            When characters message you, their conversations will appear here.
          </p>
          <button onClick={fetchDMThreads} className="text-xs text-primary hover:underline mt-3">
            Refresh
          </button>
        </div>
      ) : (
        <div className="space-y-1">
          {dmThreads.map(thread => (
            <div
              key={thread.id}
              className="w-full flex items-center gap-3 bg-card border rounded-xl p-3 hover:border-primary/30 hover:bg-muted/30 transition-all text-left"
            >
              <button
                onClick={() => selectDMThread(thread)}
                className="flex flex-1 min-w-0 items-center gap-3 text-left"
              >
                <div className="w-10 h-10 rounded-full shrink-0 bg-muted overflow-hidden relative">
                  {thread.character_avatar ? (
                    <img src={thread.character_avatar} alt="" className="w-full h-full object-cover" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-sm font-bold text-muted-foreground">
                      {thread.character_name?.[0] || '?'}
                    </div>
                  )}
                  {(thread.unread_count || 0) > 0 && (
                    <div className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-primary text-[8px] font-bold text-primary-foreground flex items-center justify-center">
                      {Math.min(thread.unread_count, 9)}
                    </div>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold truncate">{thread.character_name}</span>
                    <span className="text-[9px] text-muted-foreground shrink-0">
                      {thread.last_message?.timestamp ? formatTime(thread.last_message.timestamp) : ''}
                    </span>
                  </div>
                  {thread.last_message?.content ? (
                    <p className="text-[11px] text-muted-foreground truncate mt-0.5">
                      {thread.last_message.role === 'character' ? '' : 'You: '}
                      {thread.last_message.content}
                    </p>
                  ) : (
                    <p className="text-[11px] text-muted-foreground/50 italic mt-0.5">No messages yet</p>
                  )}
                </div>
              </button>
              <button
                onClick={() => {
                  if (window.confirm(`Delete DM thread with ${thread.character_name || 'this character'}?`)) deleteDMThread(thread.id);
                }}
                className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-muted-foreground/60 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                title="Delete DM thread"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
