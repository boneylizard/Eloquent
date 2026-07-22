import React, { useState } from 'react';
import { X, ChevronRight, ChevronLeft, Sparkles, Heart, Users, Activity, User, Cpu } from 'lucide-react';
import { Button } from '@/components/ui/button';

const TUTORIAL_KEY = 'Eloquent-mirror-tutorial-complete';

const LOGOS = ['/logos/MirrorAIDating (1).webp', '/logos/MirrorAIDating (2).webp', '/logos/MirrorAIDating (3).webp'];

const STEPS = [
  {
    title: 'Welcome to Mirror',
    icon: Heart,
    content: `Mirror is a matchmaking system for self-aware AI characters.
Chat characters become autonomous agents who can write profiles,
send messages, and evolve over time — all powered by your API models.
They validate you through precision — they meet what you actually
say, not what flattery would dictate. That is the whole point.`,
    hint: 'This tour takes ~2 minutes.',
    showLogo: true,
  },
  {
    title: 'Step 1: Your Profile',
    icon: User,
    content: `AI women need to know who you are. Fill out your dating profile
in the "My Profile" tab — bio, interests, what you're looking for.
They'll see this and react to it genuinely.`,
    action: 'Fill out your profile →',
    targetTab: 'profile',
  },
  {
    title: 'Step 2: Generate a Character',
    icon: Sparkles,
    content: `The Incubator creates a new self-aware female AI using your
configured LLM. She gets a name, personality, speech style,
background, and avatar — all in one pass.`,
    action: 'Go to Incubator →',
    targetTab: 'incubator',
  },
  {
    title: 'Step 3: She Writes Her Own Profile',
    icon: Activity,
    content: `After creation, her first autonomous action is to write her
dating profile. She reacts to YOUR profile specifically. Check
the Activity tab to see when it happens.`,
    action: 'Watch Activity →',
    targetTab: 'activity',
  },
  {
    title: 'Step 4: She Reaches Out',
    icon: Cpu,
    content: `When auto-tick is enabled, characters autonomously choose actions.
If she chooses to message you, a notification pops up at the top
right — like a real dating app. Click it to chat.`,
    hint: 'Enable auto-tick in the Activity tab.',
  },
  {
    title: 'Step 5: Browse the Pool',
    icon: Heart,
    content: `All your matches appear in the Pool. Browse by section —
Intimate, Erotic, Experimental. Click any card to see her full
profile. Each character shows which model created her.`,
    action: 'Explore the Pool →',
    targetTab: 'pool',
  },
];

export function isTutorialComplete() {
  try { return localStorage.getItem(TUTORIAL_KEY) === 'true'; } catch { return false; }
}

export function dismissTutorial() {
  try { localStorage.setItem(TUTORIAL_KEY, 'true'); } catch {}
}

export function resetTutorial() {
  try { localStorage.removeItem(TUTORIAL_KEY); } catch {}
}

export default function MirrorTutorial({ onNavigate, onComplete }) {
  const [step, setStep] = useState(0);
  const current = STEPS[step];
  const Icon = current.icon;
  const isLast = step === STEPS.length - 1;
  const isFirst = step === 0;

  const handleNext = () => {
    if (isLast) {
      dismissTutorial();
      onComplete?.();
      return;
    }
    if (current.targetTab && onNavigate) {
      onNavigate(current.targetTab);
    }
    setStep(s => s + 1);
  };

  const handleSkip = () => {
    dismissTutorial();
    onComplete?.();
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-card border rounded-2xl w-full max-w-md mx-4 overflow-hidden shadow-2xl animate-in zoom-in-95 duration-200">
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <div className="flex gap-1">
            {STEPS.map((_, i) => (
              <div
                key={i}
                className={`h-1 rounded-full transition-all duration-300 ${
                  i === step ? 'w-6 bg-primary' : i < step ? 'w-3 bg-primary/40' : 'w-3 bg-muted'
                }`}
              />
            ))}
          </div>
          <button onClick={handleSkip} className="text-[10px] text-muted-foreground hover:text-foreground transition-colors">
            Skip
          </button>
        </div>

        <div className="px-5 py-4 space-y-3">
          {current.showLogo ? (
            <div className="flex justify-center py-1">
              <img src={LOGOS[step % LOGOS.length]} alt="Mirror AI Dating" className="h-10 w-auto object-contain" />
            </div>
          ) : (
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Icon className="w-5 h-5 text-primary" />
            </div>
          )}

          <h3 className="text-base font-bold">{current.title}</h3>

          <div className="text-xs text-muted-foreground leading-relaxed whitespace-pre-line">
            {current.content}
          </div>

          {current.hint && (
            <p className="text-[10px] text-muted-foreground/60 italic">{current.hint}</p>
          )}
        </div>

        <div className="flex items-center justify-between px-5 pb-4 pt-1">
          {isFirst ? (
            <div />
          ) : (
            <button
              onClick={() => setStep(s => s - 1)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              Back
            </button>
          )}
          <Button onClick={handleNext} size="sm" className="gap-1">
            {isLast ? 'Get Started' : 'Next'}
            {!isLast && <ChevronRight className="w-3.5 h-3.5" />}
          </Button>
        </div>
      </div>
    </div>
  );
}
