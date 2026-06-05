import React, { useCallback, useRef, useState } from 'react';

import { Button } from './ui/button';

import { Label } from './ui/label';

import { Textarea } from './ui/textarea';

import { Alert, AlertDescription, AlertTitle } from './ui/alert';

import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';

import { Loader2, RotateCcw, Send, Square } from 'lucide-react';

import {

  formatApiError,

  mergeSessionMessages,

  normalizeEndpointModelId,

  pickLongerDraft,

  readCondenserSessionStream,

} from '../utils/chatlogCondenserUtils';



const DEFAULT_OPENING =

  'Produce a dense draft markdown of the full ORIGINAL_CHATLOG. Preserve every load-bearing reasoning move; compress only filler and repetition.';



/**

 * Agentic condenser session: chat with streaming dense-draft output, reset per run.

 * Stop aborts streaming only — never clears the shared source chatlog (parent inputText).

 */

export default function ChatlogCondenserAgentPanel({

  apiUrl,

  apiReady,

  inputText,

  modelName,

  setModelName,

  modelOptions,

  includeFullLogContext,

  useRag = false,

  ragDocs = [],

  onRestoreOriginalLog,

}) {

  const [sessionId, setSessionId] = useState(null);

  const [messages, setMessages] = useState([]);

  const [partialCondensed, setPartialCondensed] = useState('');

  const [draftInput, setDraftInput] = useState(DEFAULT_OPENING);

  const [busy, setBusy] = useState(false);

  const [streamingText, setStreamingText] = useState('');

  const [error, setError] = useState(null);

  const abortRef = useRef(null);

  const sendInFlightRef = useRef(false);

  const streamingTextRef = useRef('');

  const partialCondensedRef = useRef('');

  const messagesRef = useRef([]);



  streamingTextRef.current = streamingText;

  partialCondensedRef.current = partialCondensed;

  messagesRef.current = messages;



  const applySessionSnapshot = useCallback((s, { streamingDraft = '' } = {}) => {

    if (!s) return;

    setMessages((prev) => mergeSessionMessages(prev, s.messages || []));

    setPartialCondensed((prev) =>

      pickLongerDraft(prev, s.partial_condensed, streamingDraft)

    );

  }, []);



  const refreshSession = useCallback(

    async (sid, opts = {}) => {

      const res = await fetch(`${apiUrl}/memory/chatlog-condenser/session/${sid}`);

      const data = await res.json();

      if (!res.ok) throw new Error(formatApiError(data, res.statusText));

      const s = data.session;

      applySessionSnapshot(s, opts);

      return s;

    },

    [apiUrl, applySessionSnapshot]

  );



  const cancelServerStream = useCallback(

    async (sid) => {

      if (!sid) return;

      try {

        await fetch(`${apiUrl}/memory/chatlog-condenser/session/${sid}/cancel`, {

          method: 'POST',

        });

      } catch {

        /* best-effort unlock */

      }

    },

    [apiUrl]

  );



  const handleCancel = useCallback(async () => {

    const snapStream = streamingTextRef.current;

    abortRef.current?.abort();

    abortRef.current = null;

    sendInFlightRef.current = false;

    setStreamingText('');

    setBusy(false);



    if (snapStream.trim()) {

      setPartialCondensed((prev) => pickLongerDraft(prev, snapStream));

    }



    await cancelServerStream(sessionId);

    if (sessionId) {

      try {

        await refreshSession(sessionId, { streamingDraft: snapStream });

      } catch {

        /* keep local messages/draft if refresh fails after cancel */

      }

    }

  }, [sessionId, cancelServerStream, refreshSession]);



  const handleStartSession = async () => {

    if (!apiReady) {

      setError('Backend not ready.');

      return;

    }

    if (!inputText.trim()) {

      setError('Paste or load a chatlog first (shared input above).');

      return;

    }

    if (!modelName) {

      setError('Select a model.');

      return;

    }

    setBusy(true);

    setError(null);

    setStreamingText('');

    try {

      const res = await fetch(`${apiUrl}/memory/chatlog-condenser/session/start`, {

        method: 'POST',

        headers: { 'Content-Type': 'application/json' },

        body: JSON.stringify({

          text: inputText,

          model_name: normalizeEndpointModelId(modelName),

          include_full_log_context: includeFullLogContext,
          use_rag: useRag && ragDocs.length > 0,
          rag_docs: ragDocs,

        }),

      });

      const data = await res.json();

      if (!res.ok) throw new Error(formatApiError(data, res.statusText));

      setSessionId(data.session.session_id);

      setMessages(data.session.messages || []);

      setPartialCondensed(data.session.partial_condensed || '');

      const log = data.session.original_log;

      if (log && onRestoreOriginalLog) onRestoreOriginalLog(log);

    } catch (err) {

      setError(err?.message || String(err));

    } finally {

      setBusy(false);

    }

  };



  const handleReset = async () => {

    if (!sessionId) return;

    if (busy) await handleCancel();

    setBusy(true);

    setError(null);

    setStreamingText('');

    try {

      const res = await fetch(

        `${apiUrl}/memory/chatlog-condenser/session/${sessionId}/reset`,

        { method: 'POST' }

      );

      const data = await res.json();

      if (!res.ok) throw new Error(formatApiError(data, res.statusText));

      setMessages([]);

      setPartialCondensed('');

      setDraftInput(DEFAULT_OPENING);

      const log = data.session?.original_log;

      if (log && onRestoreOriginalLog) onRestoreOriginalLog(log);

    } catch (err) {

      setError(err?.message || String(err));

    } finally {

      setBusy(false);

    }

  };



  const handleSend = async () => {

    if (!sessionId || !draftInput.trim()) return;

    if (!apiReady) return;

    if (busy || sendInFlightRef.current) return;



    const userText = draftInput.trim();

    setDraftInput('');

    setBusy(true);

    sendInFlightRef.current = true;

    setError(null);

    setStreamingText('');

    setMessages((prev) => [...prev, { role: 'user', content: userText }]);



    const controller = new AbortController();

    abortRef.current = controller;

    let assistantAccum = '';

    try {

      const res = await fetch(

        `${apiUrl}/memory/chatlog-condenser/session/${sessionId}/message`,

        {

          method: 'POST',

          headers: { 'Content-Type': 'application/json' },

          body: JSON.stringify({ message: userText }),

          signal: controller.signal,

        }

      );

      await readCondenserSessionStream(res, {

        signal: controller.signal,

        onToken: (text) => {

          assistantAccum += text;

          setStreamingText(assistantAccum);

        },

        onDone: (payload) => {

          const md = payload.condensed_markdown || assistantAccum;

          if (md) {

            setPartialCondensed(md);

            setStreamingText('');

            setMessages((prev) => [...prev, { role: 'assistant', content: md }]);

          } else if (payload.interrupted) {

            setPartialCondensed((prev) => pickLongerDraft(prev, assistantAccum));

            setStreamingText('');

          }

        },

      });

      await refreshSession(sessionId, { streamingDraft: assistantAccum });

    } catch (err) {

      if (err.name === 'AbortError') {

        const snap = assistantAccum || streamingTextRef.current;

        if (snap.trim()) {

          setPartialCondensed((prev) => pickLongerDraft(prev, snap));

        }

        await cancelServerStream(sessionId);

        try {

          await refreshSession(sessionId, { streamingDraft: snap });

        } catch {

          /* keep local state */

        }

      } else {

        setError(err?.message || String(err));

        setMessages((prev) => {

          if (prev.length && prev[prev.length - 1].role === 'user') {

            return prev.slice(0, -1);

          }

          return prev;

        });

        setDraftInput(userText);

        await cancelServerStream(sessionId);

      }

      setStreamingText('');

    } finally {

      setBusy(false);

      sendInFlightRef.current = false;

      abortRef.current = null;

    }

  };



  const liveDraft = streamingText || partialCondensed;



  return (

    <div className="space-y-4">

      <p className="text-sm text-muted-foreground leading-relaxed">

        Chat with a condenser agent that always sees the original log and every draft attempt in this

        run. Ask it to continue if output cuts off, or refine the draft in follow-up messages. Reset

        clears the run and starts over with the same loaded chatlog.

      </p>



      <div className="grid gap-3 sm:grid-cols-2">

        <div className="space-y-2">

          <Label>Model</Label>

          <Select value={modelName} onValueChange={setModelName} disabled={busy}>

            <SelectTrigger>

              <SelectValue placeholder="Select model" />

            </SelectTrigger>

            <SelectContent>

              {modelOptions.map((o) => (

                <SelectItem key={o.id} value={o.id}>

                  {o.label}

                </SelectItem>

              ))}

            </SelectContent>

          </Select>

        </div>

        <div className="flex flex-wrap items-end gap-2">

          <Button

            type="button"

            variant="outline"

            onClick={handleStartSession}

            disabled={busy || !inputText.trim()}

          >

            {busy && !sessionId ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}

            Start session

          </Button>

          <Button

            type="button"

            variant="outline"

            onClick={handleReset}

            disabled={!sessionId || busy}

          >

            <RotateCcw className="mr-2 h-4 w-4" />

            Reset run

          </Button>

        </div>

      </div>



      {sessionId && (

        <p className="text-xs text-muted-foreground font-mono truncate">

          Session: {sessionId}

        </p>

      )}



      {error && (

        <Alert variant="destructive">

          <AlertTitle>Error</AlertTitle>

          <AlertDescription>{error}</AlertDescription>

        </Alert>

      )}



      <div className="rounded-xl border border-border/80 bg-card/40 p-3 space-y-3 max-h-[320px] overflow-y-auto">

        {!sessionId && (

          <p className="text-sm text-muted-foreground">Start a session to begin chatting.</p>

        )}

        {messages.map((m, i) => (

          <div

            key={`${i}-${m.role}`}

            className={

              m.role === 'user'

                ? 'text-sm rounded-lg bg-muted/60 p-2'

                : 'text-sm rounded-lg border border-border/60 p-2 font-mono text-xs whitespace-pre-wrap'

            }

          >

            <span className="font-medium text-foreground/80 block mb-1">

              {m.role === 'user' ? 'You' : 'Condenser'}

            </span>

            {m.content}

          </div>

        ))}

        {streamingText && (

          <div className="text-sm rounded-lg border border-primary/40 p-2 font-mono text-xs whitespace-pre-wrap">

            <span className="font-medium text-primary block mb-1">

              Condenser <Loader2 className="inline h-3 w-3 animate-spin ml-1" />

            </span>

            {streamingText}

          </div>

        )}

      </div>



      <div className="space-y-2">

        <Label>Dense draft (live)</Label>

        <Textarea

          className="min-h-[200px] font-mono text-xs"

          readOnly

          value={liveDraft}

          placeholder="Streaming output appears here…"

        />

      </div>



      <div className="flex gap-2">

        <Textarea

          className="min-h-[72px] flex-1 text-sm"

          value={draftInput}

          onChange={(e) => setDraftInput(e.target.value)}

          placeholder="Instruction or continuation…"

          disabled={!sessionId || busy}

          onKeyDown={(e) => {

            if (e.key === 'Enter' && !e.shiftKey) {

              e.preventDefault();

              if (!busy && draftInput.trim()) handleSend();

            }

          }}

        />

        {busy && sendInFlightRef.current ? (

          <Button

            type="button"

            variant="destructive"

            onClick={handleCancel}

            className="self-end"

            title="Stop streaming"

          >

            <Square className="h-4 w-4 fill-current" />

          </Button>

        ) : (

          <Button

            type="button"

            onClick={handleSend}

            disabled={!sessionId || !draftInput.trim() || busy}

            className="self-end"

          >

            <Send className="h-4 w-4" />

          </Button>

        )}

      </div>



      <p className="text-xs text-muted-foreground">

        API:{' '}

        <code className="px-1 rounded bg-muted">POST …/session/start</code>,{' '}

        <code className="px-1 rounded bg-muted">POST …/session/&#123;id&#125;/message</code> (SSE),{' '}

        <code className="px-1 rounded bg-muted">POST …/cancel</code>,{' '}

        <code className="px-1 rounded bg-muted">POST …/reset</code>

      </p>

    </div>

  );

}


