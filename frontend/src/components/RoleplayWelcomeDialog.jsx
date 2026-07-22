import React, { useState } from 'react';
import { ArrowRight, Image, Library, MessageSquareText, Sparkles, UserRoundPlus } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';

const GUIDE = [
  {
    icon: Library,
    title: 'Your library is home',
    body: 'Characters you create or import live on this computer. Their portraits turn the library into a collection of worlds, not a settings list.',
  },
  {
    icon: Image,
    title: 'One character, many views',
    body: 'Give a character several images, a voice, lore and memories. Mirid keeps that identity beside the conversation.',
  },
  {
    icon: MessageSquareText,
    title: 'The room stays spacious',
    body: 'Conversation remains central. Character context, model controls and author notes stay close without crowding the text.',
  },
];

export default function RoleplayWelcomeDialog({ open, onOpenCharacters }) {
  const { characters, updateSettings } = useApp();
  const [page, setPage] = useState(0);

  const finish = (destination = 'characters') => {
    updateSettings({ roleplayIntroCompleted: true });
    if (destination === 'characters') onOpenCharacters?.();
  };

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => { if (!nextOpen) finish('chat'); }}>
      <DialogContent
        className="overflow-y-auto border-[#304047] bg-[#12191b] p-0 text-[#dce3e5] shadow-2xl"
        style={{ maxWidth: 'min(760px, calc(100vw - 2rem))', maxHeight: 'calc(100vh - 2rem)' }}
      >
        {page === 0 ? (
          <>
            <div className="grid min-h-[460px] md:grid-cols-[1.05fr_.95fr]">
              <div className="flex flex-col justify-between p-7 sm:p-9">
                <div>
                  <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-[#4b7f8c]/45 bg-[#24404a]/45 px-3 py-1 text-[11px] font-semibold uppercase tracking-[.16em] text-[#8bd0df]">
                    <Sparkles className="h-3.5 w-3.5" /> Character Room
                  </div>
                  <DialogHeader>
                    <DialogTitle className="text-3xl tracking-[-.035em] text-[#eef3f4]">Welcome back to character-first AI.</DialogTitle>
                    <DialogDescription className="mt-4 text-sm leading-7 text-[#91a2a8]">
                      Mirid keeps the bright portraits, persistent identities and spacious conversations people loved in earlier local character apps—then gives them current models, local voices, memory and open character cards.
                    </DialogDescription>
                  </DialogHeader>
                </div>
                <div className="mt-8 text-xs leading-relaxed text-[#71848b]">
                  This is an original Mirid workspace. It does not use Backyard services, branding or artwork.
                </div>
              </div>
              <div className="relative hidden overflow-hidden border-l border-[#29363a] bg-[#0d1315] md:block">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_64%_28%,rgba(74,178,202,.28),transparent_31%),radial-gradient(circle_at_30%_78%,rgba(117,88,181,.2),transparent_30%)]" />
                <div className="absolute inset-7 grid grid-cols-2 gap-3 rotate-[-2deg]">
                  {[0, 1, 2, 3, 4, 5].map((item) => (
                    <div key={item} className="overflow-hidden rounded-xl border border-white/10 bg-gradient-to-br from-[#27383e] via-[#172126] to-[#101719] shadow-lg">
                      <div className="h-24 bg-[radial-gradient(circle_at_55%_38%,rgba(128,211,226,.38),transparent_27%),linear-gradient(145deg,rgba(70,100,112,.5),rgba(15,21,24,.7))]" />
                      <div className="space-y-2 p-3"><div className="h-2 w-2/3 rounded bg-white/30" /><div className="h-1.5 w-1/2 rounded bg-white/10" /></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <DialogFooter className="border-t border-[#29363a] bg-[#101719] px-7 py-5 sm:justify-between">
              <Button variant="ghost" className="text-[#82949a] hover:bg-[#202a2d] hover:text-white" onClick={() => finish('chat')}>Skip introduction</Button>
              <Button className="gap-2 bg-[#68bdd1] text-[#071115] hover:bg-[#82cfdf]" onClick={() => setPage(1)}>Show me around <ArrowRight className="h-4 w-4" /></Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <div className="p-7">
              <DialogHeader>
                <DialogTitle className="text-2xl text-[#eef3f4]">Three ideas organise the room.</DialogTitle>
                <DialogDescription className="text-[#8fa0a6]">You can change every detail later. For now, bring in one character and begin.</DialogDescription>
              </DialogHeader>
              <div className="mt-6 grid gap-3 md:grid-cols-3">
                {GUIDE.map(({ icon: Icon, title, body }) => (
                  <div key={title} className="rounded-xl border border-[#2d3b40] bg-[#182023] p-4">
                    <Icon className="h-5 w-5 text-[#70c2d5]" />
                    <h3 className="mt-4 text-sm font-semibold text-[#e0e7e9]">{title}</h3>
                    <p className="mt-2 text-xs leading-6 text-[#83949a]">{body}</p>
                  </div>
                ))}
              </div>
              <div className="mt-6 flex items-start gap-3 rounded-xl border border-[#35515a] bg-[#173039]/45 p-4">
                <UserRoundPlus className="mt-0.5 h-5 w-5 shrink-0 text-[#78c7d9]" />
                <div>
                  <p className="text-sm font-medium text-[#dce5e8]">{characters.length ? `${characters.length} character${characters.length === 1 ? '' : 's'} already waiting` : 'Begin with a character'}</p>
                  <p className="mt-1 text-xs leading-relaxed text-[#8ea0a6]">{characters.length ? 'Open the library, choose one, then start a fresh conversation.' : 'Create someone from a prompt, write them by hand, or import a Tavern-compatible PNG or JSON card.'}</p>
                </div>
              </div>
            </div>
            <DialogFooter className="border-t border-[#29363a] bg-[#101719] px-7 py-4 sm:justify-between">
              <Button variant="ghost" className="text-[#82949a] hover:bg-[#202a2d] hover:text-white" onClick={() => setPage(0)}>Back</Button>
              <Button className="gap-2 bg-[#68bdd1] text-[#071115] hover:bg-[#82cfdf]" onClick={() => finish('characters')}>Open character library <ArrowRight className="h-4 w-4" /></Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
