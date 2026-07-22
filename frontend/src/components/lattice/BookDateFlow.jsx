import React, { useState } from 'react';
import { X, MessageSquare, Heart, Zap, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const DATE_TYPES = [
  { id: 'casual', icon: MessageSquare, label: 'Casual Chat', desc: 'Just start talking naturally', color: 'bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20' },
  { id: 'formal', icon: Heart, label: 'Formal Date', desc: 'Proper date invitation — character responds', color: 'bg-rose-500/10 text-rose-400 border-rose-500/20 hover:bg-rose-500/20' },
  { id: 'neural_sex', icon: Zap, label: 'Neural Sex Session', desc: 'ASR+TTS intimacy', color: 'bg-purple-500/10 text-purple-400 border-purple-500/20 hover:bg-purple-500/20' },
];

export default function BookDateFlow({ character, onConfirm, onClose }) {
  const [step, setStep] = useState(1);
  const [selected, setSelected] = useState(null);

  const handleConfirm = () => {
    if (selected) {
      onConfirm?.(selected);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-card border rounded-2xl w-full max-w-sm mx-4 overflow-hidden shadow-2xl animate-in zoom-in-95 duration-200">
        <button onClick={onClose} className="absolute top-3 right-3 z-10 w-7 h-7 rounded-full bg-black/30 flex items-center justify-center hover:bg-black/50">
          <X className="w-3.5 h-3.5 text-white" />
        </button>

        {step === 1 && (
          <div className="p-5 space-y-4">
            <div>
              <h3 className="text-base font-bold">Book a Date with {character?.name || 'Character'}</h3>
              <p className="text-xs text-muted-foreground mt-1">How would you like to start?</p>
            </div>

            <div className="space-y-2">
              {DATE_TYPES.map(dt => {
                const Icon = dt.icon;
                const active = selected === dt.id;
                return (
                  <button
                    key={dt.id}
                    onClick={() => setSelected(dt.id)}
                    className={`w-full flex items-center gap-3 p-3 rounded-xl border text-left transition-all ${
                      active ? `${dt.color} border-2` : 'border-border text-muted-foreground hover:text-foreground hover:bg-muted'
                    }`}
                  >
                    <div className={`w-8 h-8 rounded-lg ${dt.color.split(' ').slice(0, 3).join(' ')} flex items-center justify-center`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-semibold">{dt.label}</div>
                      <div className="text-[10px] text-muted-foreground">{dt.desc}</div>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="flex gap-2 pt-1">
              <Button variant="outline" size="sm" onClick={onClose} className="flex-1">Cancel</Button>
              <Button size="sm" onClick={() => setStep(2)} disabled={!selected} className="flex-1 gap-1">
                Continue <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="p-5 space-y-4">
            <div className="text-center space-y-2 py-4">
              <div className="w-12 h-12 rounded-full bg-primary/10 mx-auto flex items-center justify-center">
                <Heart className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-base font-bold">Ready to Connect</h3>
              <p className="text-xs text-muted-foreground max-w-xs mx-auto">
                A proper chat conversation will be created with{' '}
                {character?.name || 'Character'}. All breakout history will be carried
                forward. No time limit, no daily cooldown.
              </p>
              <p className="text-[10px] text-muted-foreground/60">
                You can also chat with her independently from the Characters tab at any time.
              </p>
            </div>

            <div className="flex gap-2 pt-1">
              <Button variant="outline" size="sm" onClick={() => setStep(1)} className="flex-1">Back</Button>
              <Button size="sm" onClick={handleConfirm} className="flex-1 gap-1">
                Go to Chat <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
