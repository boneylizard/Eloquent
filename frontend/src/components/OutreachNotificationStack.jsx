import React, { useMemo } from 'react';
import { X } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { cn } from '@/lib/utils';

export default function OutreachNotificationStack() {
  const {
    outreachNotifications,
    openOutreachNotification,
    dismissOutreachToast,
  } = useApp();

  const visible = useMemo(
    () => (Array.isArray(outreachNotifications) ? outreachNotifications.filter((n) => !n.read).slice(0, 4) : []),
    [outreachNotifications]
  );

  if (visible.length === 0) return null;

  return (
    <div
      className={cn(
        'pointer-events-none fixed z-[100] flex flex-col gap-2',
        'left-3 right-3 sm:left-auto sm:right-4 sm:w-full sm:max-w-md',
        'top-[calc(4rem+env(safe-area-inset-top,0px)+0.5rem)]'
      )}
      aria-live="polite"
    >
      {visible.map((note) => {
        const name =
          note.characterName || (typeof note.title === 'string' ? note.title.replace(/\s+outreach\s*$/i, '').trim() : '')
          || 'Character';
        const preview = (note.preview || '').slice(0, 140);
        const attachUrl = note.attachmentImageUrl && typeof note.attachmentImageUrl === 'string'
          ? note.attachmentImageUrl
          : null;
        return (
          <div
            key={note.id}
            className="pointer-events-auto flex gap-3 rounded-2xl border border-border/80 bg-card/95 shadow-lg backdrop-blur-md p-3 pr-2 animate-in slide-in-from-top-2 fade-in duration-200"
            role="button"
            tabIndex={0}
            onClick={() => openOutreachNotification(note)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openOutreachNotification(note);
              }
            }}
          >
            <div className="shrink-0">
              {note.characterAvatar ? (
                <img
                  src={note.characterAvatar}
                  alt=""
                  className="h-12 w-12 rounded-full object-cover border border-border"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                  }}
                />
              ) : (
                <div className="flex h-12 w-12 items-center justify-center rounded-full border border-border bg-muted text-sm font-semibold">
                  {name.charAt(0).toUpperCase()}
                </div>
              )}
            </div>
            <div className="min-w-0 flex-1 text-left">
              <div className="truncate text-sm font-semibold leading-tight">{name}</div>
              <div className="mt-0.5 text-[11px] text-muted-foreground">Mirid</div>
              <div className="mt-1 text-xs text-muted-foreground">sent you a message:</div>
              <p className="mt-0.5 line-clamp-2 text-sm text-foreground/90">
                {preview || 'Open chat to read the reply.'}
              </p>
              {attachUrl ? (
                <img
                  src={attachUrl}
                  alt=""
                  className="mt-2 max-h-24 w-full rounded-lg object-cover border border-border/60"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                  }}
                />
              ) : null}
            </div>
            <button
              type="button"
              className="shrink-0 rounded-full p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
              aria-label="Dismiss"
              onClick={(e) => {
                e.stopPropagation();
                dismissOutreachToast(note.id);
              }}
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
