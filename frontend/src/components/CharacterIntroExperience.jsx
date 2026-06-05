import React, { useMemo } from 'react';
import { RefreshCw, Sparkles, MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import {
  CHARACTER_INTRO_UI_LABELS,
  isCharacterIntroPartial,
  isCharacterIntroReady,
} from '../utils/characterIntro';
import CharacterAvatarMedia from './CharacterAvatarMedia';
import { getActiveCharacterAvatar } from '../utils/characterAvatars';

const SECTION_ACCENTS = {
  who_they_are: 'border-cyan-500/30 bg-cyan-500/5',
  how_they_engage: 'border-fuchsia-500/30 bg-fuchsia-500/5',
  tone: 'border-amber-500/30 bg-amber-500/5',
  voice_line: 'border-emerald-500/30 bg-emerald-500/10',
};

function IntroSection({ id, label, text, className }) {
  if (!text?.trim()) return null;
  return (
    <section
      className={cn(
        'rounded-xl border p-4 shadow-sm transition-all duration-500 animate-in fade-in slide-in-from-bottom-2',
        SECTION_ACCENTS[id] || 'border-border bg-card',
        className
      )}
    >
      <h3 className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground mb-2">
        {label}
      </h3>
      <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap">{text}</p>
    </section>
  );
}

function ThemeTags({ themes }) {
  if (!themes?.length) return null;
  return (
    <div className="flex flex-wrap gap-2 justify-center">
      {themes.map((t) => (
        <span
          key={t}
          className="rounded-full border border-border bg-muted/60 px-3 py-1 text-xs font-medium text-muted-foreground"
        >
          {t}
        </span>
      ))}
    </div>
  );
}

function IntroLoadingPanel() {
  return (
    <div
      className="w-full rounded-xl border border-border/70 bg-muted/25 dark:bg-muted/15 px-6 py-10 text-center shadow-sm"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <p className="text-sm font-medium text-foreground/90 intro-reality-pulse">
        Working on your current reality…
      </p>
      <p className="mt-2 text-xs text-muted-foreground">
        Shaping who they are, how they meet you, and the tone of this chat.
      </p>
      <div className="mt-8 flex justify-center items-end gap-2 h-8" aria-hidden>
        {[0, 1, 2, 3, 4].map((i) => (
          <span
            key={i}
            className="w-1.5 rounded-full bg-primary/50 intro-reality-bar"
            style={{ animationDelay: `${i * 140}ms` }}
          />
        ))}
      </div>
      <style>{`
        @keyframes intro-reality-pulse {
          0%, 100% { opacity: 0.55; }
          50% { opacity: 1; }
        }
        @keyframes intro-reality-bar {
          0%, 100% { height: 0.5rem; opacity: 0.35; }
          50% { height: 1.75rem; opacity: 0.9; }
        }
        .intro-reality-pulse {
          animation: intro-reality-pulse 2.4s ease-in-out infinite;
        }
        .intro-reality-bar {
          height: 0.5rem;
          animation: intro-reality-bar 1.2s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}

export default function CharacterIntroExperience({
  character,
  userProfile,
  status = 'loading',
  error,
  result,
  onRegenerate,
  onRetry,
  onBegin,
  variant = 'character',
  uiLabels,
}) {
  const isSystemVariant = variant === 'system';
  const labels = uiLabels || CHARACTER_INTRO_UI_LABELS;
  const display = useMemo(() => {
    if (!isCharacterIntroReady(result)) return null;
    return result.data;
  }, [result]);

  const avatarUrl = character ? getActiveCharacterAvatar(character) : null;
  const charName = character?.name || (isSystemVariant ? 'System' : 'Character');
  const userName = userProfile?.name || userProfile?.username || 'you';
  const isLoading = status === 'loading';
  const isReady = status === 'ready';
  const isError = status === 'error';
  const isPartial = isReady && isCharacterIntroPartial(result);

  return (
    <div className="flex flex-col items-center w-full max-w-2xl mx-auto py-6 md:py-10 px-2 md:px-4 animate-in fade-in duration-500">
      <div className="relative mb-6">
        <div
          className={cn(
            'absolute -inset-3 rounded-full bg-primary/15 blur-xl transition-opacity duration-700',
            isLoading ? 'opacity-100 intro-avatar-glow' : 'opacity-80'
          )}
          aria-hidden
        />
        <div className="relative h-24 w-24 md:h-28 md:w-28 rounded-full overflow-hidden border-2 border-primary/40 shadow-lg ring-4 ring-background">
          {avatarUrl ? (
            <CharacterAvatarMedia
              url={avatarUrl}
              alt={charName}
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="h-full w-full flex items-center justify-center bg-primary text-primary-foreground text-3xl font-semibold">
              {charName.charAt(0).toUpperCase()}
            </div>
          )}
        </div>
      </div>

      <div className="text-center mb-8 space-y-2">
        <p className="text-[10px] font-semibold uppercase tracking-[0.35em] text-muted-foreground">
          {isSystemVariant ? 'About this system' : 'Character introduction'}
        </p>
        <h2 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground">
          {isSystemVariant ? charName : charName}
        </h2>
        {isReady && display?.headline ? (
          <p className="text-base md:text-lg text-muted-foreground max-w-lg mx-auto leading-snug animate-in fade-in duration-500">
            {display.headline}
          </p>
        ) : null}
        <p className="text-xs text-muted-foreground/80">
          {isSystemVariant
            ? `Overview of how this system will work with ${userName} — character card is used as the system prompt.`
            : `A personalized opening for ${userName} — replace the static greeting when enabled in Settings.`}
        </p>
      </div>

      {isPartial && (
        <p className="w-full mb-3 text-center text-xs text-amber-600 dark:text-amber-500/90 animate-in fade-in">
          Partial introduction — some sections were recovered from incomplete JSON. Regenerate for a fuller card.
        </p>
      )}

      {isError && error && (
        <div className="w-full mb-4 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive animate-in fade-in whitespace-pre-wrap">
          {error}
          <Button variant="ghost" size="sm" className="mt-2" onClick={onRetry}>
            <RefreshCw className="h-3.5 w-3.5 mr-1" />
            Try again
          </Button>
        </div>
      )}

      <div className="w-full space-y-4">
        {isLoading && <IntroLoadingPanel />}

        {isReady && display && (
          <div className="space-y-4 animate-in fade-in duration-500">
            <IntroSection
              id="who_they_are"
              label={labels.who_they_are}
              text={display.who_they_are}
            />
            <IntroSection
              id="how_they_engage"
              label={labels.how_they_engage}
              text={display.how_they_engage}
            />
            <IntroSection
              id="tone"
              label={labels.tone}
              text={display.tone}
            />

            {display.voice_line ? (
              <blockquote className="rounded-xl border border-emerald-500/25 bg-emerald-500/5 px-5 py-4 text-center">
                <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground mb-2">
                  {labels.voice_line}
                </p>
                <p className="text-base italic text-foreground leading-relaxed">
                  &ldquo;{display.voice_line}&rdquo;
                </p>
              </blockquote>
            ) : null}

            <ThemeTags themes={display.themes} />
          </div>
        )}
      </div>

      <div className="mt-10 flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
        <Button
          size="lg"
          className="w-full sm:w-auto gap-2 shadow-md"
          onClick={onBegin}
          disabled={!isReady}
        >
          <MessageCircle className="h-4 w-4" />
          Begin conversation
        </Button>
        {isReady && (
          <Button
            variant="outline"
            size="lg"
            className="w-full sm:w-auto gap-2"
            onClick={onRegenerate}
          >
            <Sparkles className="h-4 w-4" />
            Regenerate
          </Button>
        )}
      </div>

      <p className="mt-6 text-[11px] text-muted-foreground text-center max-w-md">
        Send a message below to skip ahead, or use Begin conversation to enter the chat without a canned greeting.
      </p>

      <style>{`
        @keyframes intro-avatar-glow {
          0%, 100% { opacity: 0.5; transform: scale(0.98); }
          50% { opacity: 1; transform: scale(1.02); }
        }
        .intro-avatar-glow {
          animation: intro-avatar-glow 2.8s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}
