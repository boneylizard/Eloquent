import React, { useState, useCallback, useMemo, useRef } from 'react';
import { useApp } from '../contexts/AppContext';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Label } from '@/components/ui/label';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import BookRunSettingsPanel from './BookRunSettingsPanel';
import { Loader2, X, Download, Play, Square, ScrollText, MessageSquare, Upload, Wand2, FileDown } from 'lucide-react';

const emptyChapter = () => ({ id: `ch_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`, title: '', intent: '' });

/** Matches `runBookAutomationChapter` user message text (first run index only gets preamble). */
function buildChapterUserContent(title, intent, { isFirstInRun, preamble }) {
  const titleLine = (title || '').trim();
  const intentText = (intent || '').trim();
  const body = [titleLine && `# ${titleLine}`, intentText].filter(Boolean).join('\n\n');
  if (!body.trim()) return '';
  const p = (preamble || '').trim();
  return isFirstInRun && p ? `${p}\n\n${body}` : body;
}

/** One block from a `---`-delimited file (optional first-line markdown title). */
function parseOneChapterBlock(block) {
  const lines = block.split('\n');
  const first = (lines[0] || '').trim();
  const hm = first.match(/^#{1,6}\s+(.+)$/);
  if (hm) {
    const intent = lines.slice(1).join('\n').trim();
    return { title: hm[1].trim(), intent: intent || block.trim() };
  }
  return { title: '', intent: block.trim() };
}

/**
 * Build chapter rows from .txt / .md / .json for the queue (not the same as "download book output").
 * Supports: JSON [{title, intent}], `---` separated sections (matches export delimiter), or # headings per chapter.
 */
function parseChaptersFromFileText(raw) {
  const text = String(raw || '').replace(/^\uFEFF/, '').trim();
  if (!text) return [];

  if (text.startsWith('[')) {
    try {
      const arr = JSON.parse(text);
      if (Array.isArray(arr)) {
        return arr
          .map((row) => ({
            title: String(row?.title ?? '').trim(),
            intent: String(row?.intent ?? row?.body ?? row?.prompt ?? '').trim(),
          }))
          .filter((r) => r.intent || r.title);
      }
    } catch (_) {
      return [];
    }
  }

  const byDelim = text.split(/\n-{3,}\s*\n/).map((s) => s.trim()).filter(Boolean);
  if (byDelim.length > 1) {
    return byDelim.map(parseOneChapterBlock).filter((r) => r.intent);
  }

  const out = [];
  let curTitle = '';
  const curBody = [];
  const flush = () => {
    const intent = curBody.join('\n').trim();
    if (curTitle || intent) {
      const row = { title: curTitle, intent: intent || curTitle };
      if (row.intent) out.push(row);
    }
    curTitle = '';
    curBody.length = 0;
  };

  const lines = text.split('\n');
  for (const line of lines) {
    const m = line.match(/^#{1,6}\s+(.+)$/);
    if (m) {
      flush();
      curTitle = m[1].trim();
    } else {
      curBody.push(line);
    }
  }
  flush();

  if (out.length) return out;

  return text ? [{ title: '', intent: text }] : [];
}

const BookWriterOverlay = ({ open, onClose }) => {
  const {
    primaryIsAPI,
    activeConversation,
    conversations,
    messages,
    activeCharacter,
    primaryModel,
    isGenerating,
    handleStopGeneration,
    settings,
    beginBookAutomationPacking,
    endBookAutomationPacking,
    runBookAutomationChapter,
    runBookAutomationQuickPrompt,
    buildBookAutomationExport,
    updateSettings,
    generateBookChapterJsonOutline,
  } = useApp();

  const [chapters, setChapters] = useState([emptyChapter()]);
  const [status, setStatus] = useState('');
  const [running, setRunning] = useState(false);
  const importInputRef = useRef(null);
  const outlineUploadInputRef = useRef(null);
  const [outlineNotes, setOutlineNotes] = useState('');
  const [outlineUploadText, setOutlineUploadText] = useState('');
  const [outlineUploadName, setOutlineUploadName] = useState('');
  const [outlineJsonOut, setOutlineJsonOut] = useState('');
  const [outlineErr, setOutlineErr] = useState('');
  const [outlineDirection, setOutlineDirection] = useState('');

  const quickButtons = useMemo(
    () => (Array.isArray(settings.bookQuickPromptButtons) ? settings.bookQuickPromptButtons : []).filter((b) => b && (b.label || b.text)),
    [settings.bookQuickPromptButtons]
  );

  const sessionLink = useMemo(() => {
    const conv = conversations?.find((c) => c.id === activeConversation);
    const title = (conv?.name || '').trim() || (activeConversation ? 'Untitled chat' : '');
    const n = Array.isArray(messages) ? messages.length : 0;
    const last = n > 0 ? messages[n - 1] : null;
    let lastPreview = '';
    if (last && typeof last.content === 'string' && last.content.trim()) {
      lastPreview = last.content.replace(/\s+/g, ' ').trim();
      if (lastPreview.length > 140) lastPreview = `${lastPreview.slice(0, 140)}…`;
    }
    const charName = (activeCharacter?.name || '').trim();
    const who =
      charName ||
      (settings.multiRoleMode ? 'Multi-role roster (speaker resolved per turn)' : 'Default assistant voice');
    const lastRoleLabel =
      last?.role === 'user' ? 'user' : last?.role === 'bot' ? 'assistant' : 'message';
    return {
      hasChat: Boolean(activeConversation),
      title,
      messageCount: n,
      lastRoleLabel,
      lastPreview,
      who,
    };
  }, [conversations, activeConversation, messages, activeCharacter?.name, settings.multiRoleMode]);

  /** First non-empty-instructions chapter = first queued step (same rule as Start run). */
  const firstOutgoingUserMessage = useMemo(() => {
    const preamble = settings.bookWordFloorPreamble || '';
    for (let i = 0; i < chapters.length; i += 1) {
      const intent = (chapters[i].intent || '').trim();
      if (!intent) continue;
      return buildChapterUserContent(chapters[i].title, chapters[i].intent, {
        isFirstInRun: true,
        preamble,
      });
    }
    return '';
  }, [chapters, settings.bookWordFloorPreamble]);

  /** Queue position (0-based) for each row; null = skipped until instructions filled. */
  const chapterQueueIndex = useMemo(() => {
    const out = chapters.map(() => null);
    let n = 0;
    for (let i = 0; i < chapters.length; i += 1) {
      if (!(chapters[i].intent || '').trim()) continue;
      out[i] = n;
      n += 1;
    }
    return out;
  }, [chapters]);

  const updateChapter = useCallback((id, patch) => {
    setChapters((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  }, []);

  const addChapter = useCallback(() => {
    setChapters((prev) => [...prev, emptyChapter()]);
  }, []);

  const removeChapter = useCallback((id) => {
    setChapters((prev) => (prev.length <= 1 ? prev : prev.filter((c) => c.id !== id)));
  }, []);

  const downloadExport = useCallback(() => {
    const body = buildBookAutomationExport();
    if (!body.trim()) {
      alert(
        'Nothing to download yet.\n\nThis button saves the assistant’s prose from chapters you have already run (joined with ---). It does not save the outline boxes below.'
      );
      return;
    }
    const blob = new Blob([body], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `book-output-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [buildBookAutomationExport]);

  const runQueue = useCallback(async () => {
    if (!primaryIsAPI) {
      alert('Book run needs an API (subscription) primary model with expanded context.');
      return;
    }
    if (!activeConversation) {
      alert('Open or start a chat first.');
      return;
    }
    const rows = chapters.map((c) => ({
      title: (c.title || '').trim(),
      intent: (c.intent || '').trim(),
    }));
    const valid = rows.filter((r) => r.intent);
    if (!valid.length) {
      alert('Add instructions in at least one chapter (the large text box — required for each chapter you want to run).');
      return;
    }

    setRunning(true);
    beginBookAutomationPacking();
    let halted = false;
    try {
      for (let i = 0; i < valid.length; i += 1) {
        const { title, intent } = valid[i];
        setStatus(`Chapter ${i + 1} / ${valid.length}${title ? `: ${title}` : ''}`);
        const res = await runBookAutomationChapter({
          chapterIndex: i,
          title,
          intent,
          isFirstInRun: i === 0,
        });
        if (res.cancelled) {
          setStatus('Cancelled.');
          halted = true;
          break;
        }
        if (!res.ok) {
          alert(`Book run stopped: ${res.error || 'Unknown error'}`);
          setStatus('Halted.');
          halted = true;
          break;
        }
      }
      if (!halted) {
        setStatus('Done.');
      }
    } finally {
      endBookAutomationPacking();
      setRunning(false);
    }
  }, [
    activeConversation,
    beginBookAutomationPacking,
    chapters,
    endBookAutomationPacking,
    primaryIsAPI,
    runBookAutomationChapter,
  ]);

  const onImportChaptersPick = useCallback(() => {
    importInputRef.current?.click();
  }, []);

  const onImportChaptersFile = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      e.target.value = '';
      if (!file || running) return;
      const reader = new FileReader();
      reader.onload = () => {
        const parsed = parseChaptersFromFileText(reader.result);
        if (!parsed.length) {
          window.alert(
            'No chapters found in that file.\n\nTry:\n• One # heading per chapter, then body text under it\n• Or sections separated by a line with only ---\n• Or JSON: [{"title":"…","intent":"…"}, …]'
          );
          return;
        }
        const ok = window.confirm(
          `Import ${parsed.length} chapter row(s) from "${file.name}"?\n\nThis replaces the chapter boxes below (your chat is not changed until you run).`
        );
        if (!ok) return;
        setChapters(parsed.map((p) => ({ ...emptyChapter(), title: p.title, intent: p.intent })));
      };
      reader.onerror = () => window.alert('Could not read that file.');
      reader.readAsText(file);
    },
    [running]
  );

  const onOutlineUploadPick = useCallback(() => {
    outlineUploadInputRef.current?.click();
  }, []);

  const onOutlineUploadFile = useCallback((e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file || running) return;
    const reader = new FileReader();
    reader.onload = () => {
      setOutlineUploadText(String(reader.result || ''));
      setOutlineUploadName(file.name);
      setOutlineErr('');
    };
    reader.onerror = () => setOutlineErr('Could not read that file.');
    reader.readAsText(file);
  }, [running]);

  const onGenerateOutlineAi = useCallback(async () => {
    if (running || isGenerating) return;
    if (!primaryIsAPI || !activeConversation) {
      alert('Needs API primary and an active conversation (same as book run).');
      return;
    }
    setOutlineErr('');
    beginBookAutomationPacking();
    try {
      const res = await generateBookChapterJsonOutline(outlineNotes, outlineUploadText, outlineDirection);
      if (!res.ok) {
        setOutlineErr(res.error || 'Generation failed.');
        setOutlineJsonOut(res.raw != null ? String(res.raw) : '');
        return;
      }
      setOutlineJsonOut(JSON.stringify(res.chapters, null, 2));
    } finally {
      endBookAutomationPacking();
    }
  }, [
    running,
    isGenerating,
    activeConversation,
    primaryIsAPI,
    outlineNotes,
    outlineUploadText,
    outlineDirection,
    beginBookAutomationPacking,
    endBookAutomationPacking,
    generateBookChapterJsonOutline,
  ]);

  const onApplyOutlineJson = useCallback(() => {
    const t = outlineJsonOut.trim();
    if (!t) {
      alert('No JSON to apply.');
      return;
    }
    let rows;
    try {
      rows = JSON.parse(t);
    } catch (e) {
      alert(`Invalid JSON: ${e.message}`);
      return;
    }
    if (!Array.isArray(rows) || !rows.length) {
      alert('JSON must be a non-empty array of { title, intent }.');
      return;
    }
    const parsed = rows
      .map((row) => ({
        title: String(row?.title ?? '').trim(),
        intent: String(row?.intent ?? row?.body ?? row?.prompt ?? '').trim(),
      }))
      .filter((r) => r.intent || r.title);
    if (!parsed.length) {
      alert('No usable chapter rows (need intent or title).');
      return;
    }
    if (!window.confirm(`Replace chapter boxes with ${parsed.length} row(s) from the JSON editor?`)) return;
    setChapters(parsed.map((p) => ({ ...emptyChapter(), title: p.title, intent: p.intent })));
  }, [outlineJsonOut]);

  const onDownloadOutlineJson = useCallback(() => {
    const t = outlineJsonOut.trim();
    if (!t) {
      alert('Nothing to download.');
      return;
    }
    const blob = new Blob([t], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `book-chapter-outline-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [outlineJsonOut]);

  const onQuick = useCallback(
    async (text) => {
      if (!text?.trim() || running) return;
      if (!primaryIsAPI || !activeConversation) {
        alert('Needs API chat and an active conversation.');
        return;
      }
      beginBookAutomationPacking();
      setRunning(true);
      setStatus('Quick prompt…');
      try {
        const res = await runBookAutomationQuickPrompt(text.trim());
        if (!res.ok && !res.cancelled) {
          alert(res.error || 'Quick prompt failed.');
        }
        setStatus(res.ok ? 'Quick prompt done.' : res.cancelled ? 'Cancelled.' : 'Done.');
      } finally {
        endBookAutomationPacking();
        setRunning(false);
      }
    },
    [activeConversation, beginBookAutomationPacking, endBookAutomationPacking, primaryIsAPI, runBookAutomationQuickPrompt, running]
  );

  if (!open) return null;

  /** While the model is working, dock to the bottom so the chat stays visible and interactive above. */
  const dockedForRun = running || isGenerating;

  return (
    <div
      className={cn(
        'fixed inset-0 z-[80] min-h-0',
        dockedForRun
          ? 'pointer-events-none flex items-end justify-stretch'
          : 'flex flex-col border-b border-border bg-background text-foreground'
      )}
    >
      <div
        className={cn(
          'flex min-h-0 flex-col overflow-hidden',
          dockedForRun
            ? 'pointer-events-auto max-h-[min(52vh,580px)] w-full rounded-t-xl border-x border-t border-primary/40 bg-background text-foreground shadow-[0_-12px_48px_rgba(0,0,0,0.18)] dark:shadow-[0_-12px_48px_rgba(0,0,0,0.5)]'
            : 'min-h-0 flex-1'
        )}
      >
      <header className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border bg-card px-4 py-3 shadow-sm">
        <ScrollText className="h-5 w-5 shrink-0 text-primary" aria-hidden />
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-base font-semibold tracking-tight text-foreground">Book chapter queue</h2>
          <p className="mt-0.5 text-sm leading-snug text-foreground/80">
            {dockedForRun ? (
              <>
                <span className="font-semibold text-foreground">{status || 'Working…'}</span>
                <span className="mt-1 block font-normal text-muted-foreground">
                  The chat above this panel updates as each step finishes — scroll and interact there while the run
                  continues.
                </span>
              </>
            ) : (
              <>Chapters are appended as real messages in the chat below — not a separate document or session.</>
            )}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {!running && !isGenerating ? (
            <Button type="button" size="sm" onClick={runQueue} disabled={isGenerating || !primaryIsAPI || !activeConversation}>
              <Play className="h-4 w-4 mr-1" />
              Start run
            </Button>
          ) : (
            <Button
              type="button"
              variant="destructive"
              size="sm"
              onClick={() => {
                handleStopGeneration();
              }}
            >
              <Square className="h-4 w-4 mr-1" />
              Stop
            </Button>
          )}
          <Button type="button" variant="ghost" size="icon" onClick={onClose} disabled={running || isGenerating} title="Close">
            <X className="h-4 w-4" />
          </Button>
        </div>
      </header>

      <section
        className={`shrink-0 border-b px-4 py-3 ${
          sessionLink.hasChat
            ? 'border-primary/30 bg-primary/[0.07] dark:bg-primary/10'
            : 'border-amber-500/40 bg-amber-500/[0.08] dark:bg-amber-950/40'
        }`}
        aria-label="Book run is tied to this chat"
      >
        <div className="mx-auto flex max-w-4xl flex-wrap items-start gap-3">
          <MessageSquare
            className={`mt-0.5 h-6 w-6 shrink-0 ${sessionLink.hasChat ? 'text-primary' : 'text-amber-600 dark:text-amber-400'}`}
            aria-hidden
          />
          <div className="min-w-0 flex-1 space-y-1.5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-foreground/70">
              {sessionLink.hasChat ? 'Continues this conversation' : 'No conversation selected'}
            </p>
            {sessionLink.hasChat ? (
              <>
                <p className="truncate text-base font-semibold text-foreground" title={sessionLink.title}>
                  {sessionLink.title}
                </p>
                <p className="text-sm text-muted-foreground">
                  <span className="text-foreground/90">{sessionLink.messageCount}</span> message
                  {sessionLink.messageCount === 1 ? '' : 's'} in thread — model sees full history when each chapter runs.
                  {primaryModel ? (
                    <>
                      {' '}
                      · Primary: <span className="text-foreground/85">{primaryModel}</span>
                      {primaryIsAPI ? '' : ' (book run needs API primary)'}
                    </>
                  ) : null}
                </p>
                <p className="text-sm text-muted-foreground">
                  Voice / character context: <span className="font-medium text-foreground/90">{sessionLink.who}</span>
                </p>
                {sessionLink.lastPreview ? (
                  <p className="border-l-2 border-primary/40 pl-3 text-xs leading-relaxed text-muted-foreground">
                    <span className="font-medium text-foreground/75">Latest {sessionLink.lastRoleLabel} turn: </span>
                    {sessionLink.lastPreview}
                  </p>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Tip: write a normal back-and-forth in chat first so the first chapter has tone and facts to build on.
                  </p>
                )}
              </>
            ) : (
              <p className="text-sm text-amber-900 dark:text-amber-100/90">
                Open a chat in the sidebar (or create one), then open this overlay again — chapters only attach to the
                currently active thread.
              </p>
            )}
          </div>
        </div>
      </section>

      <Tabs defaultValue="chapters" className="flex min-h-0 flex-1 flex-col bg-background">
        <div className="flex shrink-0 items-center border-b border-border bg-muted px-4 py-2">
          <TabsList className="bg-background/80">
            <TabsTrigger value="chapters">Chapters</TabsTrigger>
            <TabsTrigger value="settings">Run settings</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="chapters" className="mt-0 flex min-h-0 flex-1 flex-col bg-background p-0 focus-visible:ring-0">
          <input
            ref={importInputRef}
            type="file"
            accept=".txt,.md,.json,text/plain,application/json"
            className="sr-only"
            tabIndex={-1}
            aria-hidden
            onChange={onImportChaptersFile}
          />
          <input
            ref={outlineUploadInputRef}
            type="file"
            accept=".txt,.md,text/plain"
            className="sr-only"
            tabIndex={-1}
            aria-hidden
            onChange={onOutlineUploadFile}
          />

          <div className="flex min-h-0 flex-1 flex-col overflow-hidden md:flex-row">
            {/* Left: chapter editors + run controls (uses horizontal space on wide screens) */}
            <div className="flex min-h-0 min-w-0 flex-1 flex-col border-border md:border-r">
              <div className="flex shrink-0 flex-wrap gap-2 border-b border-border bg-muted px-3 py-2 md:px-4">
                <span className="mr-1 self-center text-sm font-medium text-foreground">Quick prompts</span>
                {quickButtons.length > 0 ? (
                  quickButtons.map((b) => (
                    <Button
                      key={b.id || b.label}
                      type="button"
                      size="sm"
                      variant="secondary"
                      disabled={running}
                      onClick={() => onQuick(b.text)}
                    >
                      {(b.label || 'Prompt').trim() || 'Prompt'}
                    </Button>
                  ))
                ) : (
                  <span className="self-center text-sm italic text-foreground/70">
                    None configured — add under Run settings
                  </span>
                )}
              </div>

              <div className="flex min-h-[36px] shrink-0 items-center gap-2 border-b border-border bg-card px-3 py-2 text-sm text-foreground md:px-4">
                {running && <Loader2 className="h-4 w-4 shrink-0 animate-spin text-foreground" aria-hidden />}
                <span className="min-w-0 truncate font-medium">
                  {status || (primaryIsAPI ? 'Ready.' : 'API primary model required.')}
                </span>
              </div>

              <ScrollArea className="min-h-0 flex-1 bg-background">
                <div className="space-y-4 p-3 pb-10 md:p-4">
                  {chapters.map((c, idx) => {
                    const qi = chapterQueueIndex[idx];
                    const preamble = settings.bookWordFloorPreamble || '';
                    const outgoingPreview =
                      qi !== null
                        ? buildChapterUserContent(c.title, c.intent, {
                            isFirstInRun: qi === 0,
                            preamble,
                          })
                        : '';
                    return (
                      <div key={c.id} className="space-y-3 rounded-lg border border-border bg-card p-4 shadow-sm">
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <div className="min-w-0">
                            <Label className="text-sm font-semibold text-foreground">Chapter box {idx + 1}</Label>
                            <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                              {qi === null
                                ? 'Skipped when you run — add instructions in the large field to include this step.'
                                : `Run order ${qi + 1}: inserted as your user message, then the model replies in the same chat.`}
                            </p>
                          </div>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-7 shrink-0 text-xs"
                            onClick={() => removeChapter(c.id)}
                            disabled={chapters.length <= 1 || running}
                          >
                            Remove
                          </Button>
                        </div>
                        <div className="space-y-1">
                          <Label htmlFor={`book-ch-title-${c.id}`} className="text-xs font-medium text-foreground">
                            Optional heading (markdown <code className="text-[10px]"># …</code> in chat)
                          </Label>
                          <Input
                            id={`book-ch-title-${c.id}`}
                            placeholder="e.g. Arrival at the coast"
                            value={c.title}
                            onChange={(e) => updateChapter(c.id, { title: e.target.value })}
                            disabled={running}
                            className="text-sm"
                          />
                        </div>
                        <div className="space-y-1">
                          <Label htmlFor={`book-ch-intent-${c.id}`} className="text-xs font-medium text-foreground">
                            Instructions <span className="text-destructive">*</span>{' '}
                            <span className="font-normal text-muted-foreground">(required for this step to run)</span>
                          </Label>
                          <Textarea
                            id={`book-ch-intent-${c.id}`}
                            placeholder="Write what you want the model to do in this chapter — this entire block becomes your next user message in the chat (plus optional heading above)."
                            value={c.intent}
                            onChange={(e) => updateChapter(c.id, { intent: e.target.value })}
                            disabled={running}
                            className="min-h-[120px] resize-y text-sm"
                          />
                        </div>
                        {outgoingPreview ? (
                          <div className="space-y-1">
                            <p className="text-xs font-medium text-foreground/80">Exact user message for this step</p>
                            <pre className="max-h-36 overflow-auto whitespace-pre-wrap rounded-md border border-dashed border-border bg-muted/40 p-3 font-mono text-[11px] leading-relaxed text-foreground">
                              {outgoingPreview}
                            </pre>
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                  <Button type="button" variant="outline" size="sm" onClick={addChapter} disabled={running}>
                    Add chapter box
                  </Button>
                </div>
              </ScrollArea>
            </div>

            {/* Right: import / outline / help — own scroll so it does not stack above chapter list on md+ */}
            <div className="flex max-h-[min(42vh,320px)] w-full shrink-0 flex-col overflow-hidden border-t border-border bg-muted/50 md:max-h-none md:w-[min(420px,40vw)] md:min-w-[280px] md:border-l md:border-t-0">
              <ScrollArea className="min-h-0 flex-1">
                <div className="space-y-3 px-3 py-3 md:px-4 md:py-4">
            <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
              <h3 className="text-sm font-semibold text-foreground">Save / load this outline</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                <span className="font-medium text-foreground/90">Download book output</span> — only the{' '}
                <em>assistant</em> text from chapter runs you have already finished in this chat, stitched with{' '}
                <code className="text-xs">---</code> between parts. Use it as a manuscript snapshot, not to restore the
                boxes below.
              </p>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                <span className="font-medium text-foreground/90">Import chapter outline</span> — fills the chapter
                boxes from a file (see formats under the buttons). Your chat history is unchanged until you press Start
                run.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button type="button" variant="outline" size="sm" onClick={downloadExport} disabled={running}>
                  <Download className="h-4 w-4 mr-1" aria-hidden />
                  Download book output
                </Button>
                <Button type="button" variant="outline" size="sm" onClick={onImportChaptersPick} disabled={running}>
                  <Upload className="h-4 w-4 mr-1" aria-hidden />
                  Import chapter outline…
                </Button>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                Import formats: markdown with a <code className="text-[10px]"># Chapter title</code> line per section;
                or blocks split by a line containing only <code className="text-[10px]">---</code> (same delimiter as
                download); or JSON{' '}
                <code className="text-[10px]">[{`{ "title": "…", "intent": "…" }`}]</code>.
              </p>
            </div>

            <div className="rounded-lg border border-primary/25 bg-primary/[0.06] p-4 dark:bg-primary/10">
              <h3 className="text-sm font-semibold text-foreground">What you are writing here</h3>
              <ol className="mt-2 list-inside list-decimal space-y-1.5 text-sm leading-relaxed text-foreground/85">
                <li>
                  Each chapter box defines the <strong>next user message</strong> the app will insert into your open
                  chat, then the model writes one long reply.
                </li>
                <li>
                  Optional title becomes a markdown <code className="text-xs"># Title</code> line above your
                  instructions.
                </li>
                <li>
                  The first queued chapter can also prepend the word-floor preamble from{' '}
                  <span className="font-medium">Run settings</span> (see there).
                </li>
              </ol>
              <div className="mt-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-foreground/70">
                  First user message preview (what Start run sends first)
                </p>
                {firstOutgoingUserMessage ? (
                  <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-background p-3 font-mono text-xs leading-relaxed text-foreground">
                    {firstOutgoingUserMessage}
                  </pre>
                ) : (
                  <p className="mt-1 text-sm text-muted-foreground">
                    Type instructions in the first large text box below — that text is the body of the first user
                    message.
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground">
                <Wand2 className="h-4 w-4 shrink-0 text-primary" aria-hidden />
                AI: build chapter import JSON
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                Uses the <span className="font-medium text-foreground/90">same /generate path as chat</span> (active
                character, user profile, summary, document context, rolling memory) plus a strict JSON-only task.
                Nothing is posted to the chat — only the boxes below can be filled from the result. You can add a{' '}
                <span className="font-medium text-foreground/90">custom direction</span> so the model reshapes notes
                toward a scenario (e.g. 'horror pivot', 'YA shorter chapters', 'add B-plot romance').
              </p>
              <div className="mt-3 space-y-2">
                <Label htmlFor="book-outline-direction" className="text-xs font-medium text-foreground">
                  Purpose / scenario / direction (optional but powerful)
                </Label>
                <Textarea
                  id="book-outline-direction"
                  placeholder="e.g. Re-pace for slower burn; emphasize political intrigue; keep chapters under 2k words of instructions each; rewrite for comedic tone…"
                  value={outlineDirection}
                  onChange={(e) => setOutlineDirection(e.target.value)}
                  disabled={running || isGenerating}
                  className="min-h-[72px] resize-y text-sm"
                />
              </div>
              <div className="mt-3 space-y-2">
                <Label htmlFor="book-outline-notes" className="text-xs font-medium text-foreground">
                  Notes / rough outline (optional if upload or direction is enough)
                </Label>
                <Textarea
                  id="book-outline-notes"
                  placeholder="Paste working notes, tone, arc, or bullet chapter ideas…"
                  value={outlineNotes}
                  onChange={(e) => setOutlineNotes(e.target.value)}
                  disabled={running || isGenerating}
                  className="min-h-[88px] resize-y text-sm"
                />
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={onOutlineUploadPick}
                  disabled={running || isGenerating}
                >
                  <Upload className="h-4 w-4 mr-1" aria-hidden />
                  Upload .txt…
                </Button>
                {outlineUploadName ? (
                  <span className="text-xs text-muted-foreground">
                    Loaded: <span className="font-medium text-foreground">{outlineUploadName}</span> (
                    {outlineUploadText.length.toLocaleString()} chars){' '}
                    <button
                      type="button"
                      className="ml-1 text-primary underline"
                      onClick={() => {
                        setOutlineUploadText('');
                        setOutlineUploadName('');
                      }}
                    >
                      Clear file
                    </button>
                  </span>
                ) : null}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  type="button"
                  size="sm"
                  onClick={onGenerateOutlineAi}
                  disabled={
                    running ||
                    isGenerating ||
                    !primaryIsAPI ||
                    !activeConversation ||
                    (!outlineNotes.trim() && !outlineUploadText.trim() && !outlineDirection.trim())
                  }
                >
                  {(running || isGenerating) && <Loader2 className="h-4 w-4 mr-1 animate-spin" aria-hidden />}
                  {!(running || isGenerating) && <Wand2 className="h-4 w-4 mr-1" aria-hidden />}
                  Generate JSON outline
                </Button>
                <Button type="button" variant="outline" size="sm" onClick={onDownloadOutlineJson} disabled={!outlineJsonOut.trim()}>
                  <FileDown className="h-4 w-4 mr-1" aria-hidden />
                  Save JSON file
                </Button>
                <Button type="button" variant="secondary" size="sm" onClick={onApplyOutlineJson} disabled={!outlineJsonOut.trim() || running}>
                  Apply JSON to chapter boxes
                </Button>
              </div>
              {outlineErr ? (
                <p className="mt-2 text-sm text-destructive">{outlineErr}</p>
              ) : null}
              <div className="mt-3 space-y-1">
                <Label htmlFor="book-outline-json" className="text-xs font-medium text-foreground">
                  Result (edit before apply if the model wrapped extra text)
                </Label>
                <Textarea
                  id="book-outline-json"
                  placeholder='[{"title":"…","intent":"…"}, …]'
                  value={outlineJsonOut}
                  onChange={(e) => setOutlineJsonOut(e.target.value)}
                  disabled={running}
                  className="min-h-[140px] resize-y font-mono text-xs"
                />
              </div>
            </div>
                </div>
              </ScrollArea>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="settings" className="mt-0 min-h-0 flex-1 overflow-y-auto bg-background p-4 pb-10 focus-visible:ring-0">
          <BookRunSettingsPanel
            settings={settings}
            updateSettings={updateSettings}
            disabled={running}
          />
        </TabsContent>
      </Tabs>
      </div>
    </div>
  );
};

export default BookWriterOverlay;
