import React, { useCallback, useMemo, useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { fetchWithTimeout } from '../config/api';
import {
  CHUNK_TOKENS_DEFAULT,
  CHARS_PER_TOKEN,
  formatApiError,
  normalizeCondenseParams,
  normalizeEndpointModelId,
  transcriptFromMessages,
} from '../utils/chatlogCondenserUtils';
import ChatlogCondenserAgentPanel from './ChatlogCondenserAgentPanel';
import ChatlogCondenserOrchestratorPanel from './ChatlogCondenserOrchestratorPanel';
import ChatlogCondenserRagOptions from './ChatlogCondenserRagOptions';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Input } from './ui/input';
import { Checkbox } from './ui/checkbox';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Slider } from './ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Loader2, Copy, FileUp, ScrollText, CheckCircle2 } from 'lucide-react';

function estimateTokensFromText(text) {
  return Math.max(1, Math.floor(String(text || '').length / CHARS_PER_TOKEN));
}

/** Rough chunk count before parse (even token spread); exact count comes from /parse. */
function roughChunkCountFromTokens(tokensEst, chunkTokens) {
  const budget = Math.max(1, Math.round(Number(chunkTokens) || CHUNK_TOKENS_DEFAULT));
  return Math.max(1, Math.ceil(tokensEst / budget));
}

function estimateLlmPasses(chunkCount, { runEval }) {
  const n = Math.max(1, chunkCount || 1);
  let passes = n * 2;
  if (n > 1) passes += 1;
  if (runEval) passes += 2;
  return passes;
}

/**

 * Settings → Chatlog condenser.
 * Lossless-on-meaning compression for feeding long intellectual exchanges to another model.
 */
export default function ChatlogCondenserPanel() {
  const {
    MEMORY_API_URL,
    portsReady,
    storageHydrated,
    conversations = [],
    activeConversation,
    primaryModel,
    primaryIsAPI,
    loadedModels = [],
    settings = {},
  } = useApp();

  const apiReady = portsReady && storageHydrated;
  const apiUrl = MEMORY_API_URL;

  const [inputText, setInputText] = useState('');
  const [modelName, setModelName] = useState('');
  const [targetRatio, setTargetRatio] = useState(0.4);
  const [chunkTokens, setChunkTokens] = useState(CHUNK_TOKENS_DEFAULT);
  const [overlapTurns, setOverlapTurns] = useState(5);
  const [runEval, setRunEval] = useState(true);
  const [includeFullLogContext, setIncludeFullLogContext] = useState(true);
  const [useCondenserRag, setUseCondenserRag] = useState(false);
  const [condenserRagDocs, setCondenserRagDocs] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [parsePreview, setParsePreview] = useState(null);

  const modelOptions = useMemo(() => {
    const opts = [];
    const endpoints = settings.customApiEndpoints || [];
    for (const ep of endpoints) {
      if (!ep?.enabled || !ep?.id) continue;
      opts.push({
        id: normalizeEndpointModelId(ep.id),
        label: `${ep.name || 'API'} (long-context recommended)`,
        isApi: true,
      });
    }
    for (const m of loadedModels) {
      const name = m?.name || m?.model_name;
      if (!name || String(name).startsWith('endpoint-')) continue;
      opts.push({ id: name, label: `${name} (local)`, isApi: false });
    }
    return opts;
  }, [settings.customApiEndpoints, loadedModels]);

  React.useEffect(() => {
    if (modelName) return;
    if (primaryIsAPI && primaryModel) {
      setModelName(normalizeEndpointModelId(primaryModel));
      return;
    }
    if (modelOptions.length) setModelName(modelOptions[0].id);
  }, [modelName, primaryIsAPI, primaryModel, modelOptions]);

  const activeConv = useMemo(
    () => (conversations || []).find((c) => c.id === activeConversation),
    [conversations, activeConversation]
  );

  const loadActiveConversation = () => {
    if (!activeConv?.messages?.length) {
      setError('Active conversation has no messages.');
      return;
    }
    setInputText(transcriptFromMessages(activeConv.messages));
    setError(null);
  };

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setInputText(String(reader.result || ''));
      setError(null);
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const condenseParams = useMemo(
    () =>
      normalizeCondenseParams({
        targetRatio,
        chunkTokens,
        overlapTurns,
      }),
    [targetRatio, chunkTokens, overlapTurns]
  );

  const chunkPlan = useMemo(() => {
    const tokensEst = parsePreview?.tokens_est ?? estimateTokensFromText(inputText);
    const parseMatchesBudget =
      parsePreview &&
      parsePreview._chunk_target_tokens === condenseParams.chunk_target_tokens &&
      parsePreview._overlap_turns === condenseParams.overlap_turns;
    const exactChunks = parseMatchesBudget ? parsePreview.estimated_chunk_count : null;
    const chunkCount =
      typeof exactChunks === 'number' && exactChunks > 0
        ? exactChunks
        : inputText.trim()
          ? roughChunkCountFromTokens(tokensEst, condenseParams.chunk_target_tokens)
          : null;
    if (chunkCount == null) return null;
    return {
      chunkCount,
      llmPasses: estimateLlmPasses(chunkCount, { runEval }),
      tokensEst,
      isExact: parseMatchesBudget && typeof exactChunks === 'number' && exactChunks > 0,
    };
  }, [parsePreview, inputText, condenseParams, runEval]);

  const handleParsePreview = async () => {
    if (!apiReady || !inputText.trim()) return;
    setError(null);
    const params = normalizeCondenseParams({
      targetRatio,
      chunkTokens,
      overlapTurns,
    });
    try {
      const res = await fetchWithTimeout(
        `${apiUrl}/memory/chatlog-condenser/parse`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: inputText,
            chunk_target_tokens: params.chunk_target_tokens,
            overlap_turns: params.overlap_turns,
            run_eval: runEval,
          }),
        },
        60000
      );
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data, res.statusText));
      setParsePreview({
        ...data,
        _chunk_target_tokens: params.chunk_target_tokens,
        _overlap_turns: params.overlap_turns,
      });
    } catch (err) {
      setError(err?.message || String(err));
    }
  };

  const handleCondense = async () => {
    if (!apiReady) {
      setError('Backend not ready.');
      return;
    }
    if (!inputText.trim()) {
      setError('Paste or load a chatlog first.');
      return;
    }
    if (!modelName) {
      setError('Select a model (prefer a long-context API endpoint).');
      return;
    }

    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const params = normalizeCondenseParams({
        targetRatio,
        chunkTokens,
        overlapTurns,
      });
      setTargetRatio(params.target_ratio);
      setChunkTokens(params.chunk_target_tokens);
      setOverlapTurns(params.overlap_turns);

      const res = await fetchWithTimeout(
        `${apiUrl}/memory/chatlog-condenser/condense`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: inputText,
            model_name: normalizeEndpointModelId(modelName),
            ...params,
            run_eval: runEval,
            include_full_log_context: includeFullLogContext,
            use_rag: useCondenserRag && condenserRagDocs.length > 0,
            rag_docs: condenserRagDocs,
          }),
        },
        3600000
      );
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data, res.statusText));
      setResult(data);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setBusy(false);
    }
  };

  const copyOutput = useCallback(() => {
    const md = result?.condensed_markdown;
    if (!md) return;
    navigator.clipboard?.writeText(md);
  }, [result]);

  const fidelity = result?.eval?.summary?.fidelity_score;

  return (
    <div className="space-y-4 max-w-4xl">
      <ChatlogCondenserRagOptions
        useRag={useCondenserRag}
        onUseRagChange={setUseCondenserRag}
        selectedDocs={condenserRagDocs}
        onSelectedDocsChange={setCondenserRagDocs}
      />

      <Tabs defaultValue="quick" className="w-full">
        <TabsList className="mb-4">
          <TabsTrigger value="quick">Quick condense</TabsTrigger>
          <TabsTrigger value="agent">Agent session</TabsTrigger>
          <TabsTrigger value="autonomous">Autonomous run</TabsTrigger>
        </TabsList>

        <TabsContent value="autonomous" className="mt-0">
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground leading-relaxed">
              <strong>RAG + autonomous:</strong> upload the transcript under Settings → Documents, enable
              &quot;Supplement with document context&quot; above, select that file, then start the run.
              Turn order still comes from the pasted log; RAG only fills distant callbacks when full-log
              context is off.
            </p>
            <div className="space-y-2">
              <Label>Input chatlog (shared)</Label>
              <Textarea
                className="min-h-[120px] font-mono text-xs"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="**User:** ...&#10;&#10;**Assistant:** ..."
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={loadActiveConversation}
                disabled={!activeConv}
              >
                <ScrollText className="mr-2 h-4 w-4" />
                Load active chat
              </Button>
              <Button type="button" variant="outline" size="sm" asChild>
                <label className="cursor-pointer">
                  <FileUp className="mr-2 h-4 w-4" />
                  Upload .txt / .md
                  <input
                    type="file"
                    accept=".txt,.md,.markdown"
                    className="hidden"
                    onChange={handleFile}
                  />
                </label>
              </Button>
            </div>
            <ChatlogCondenserOrchestratorPanel
              apiUrl={apiUrl}
              apiReady={apiReady}
              inputText={inputText}
              includeFullLogContext={includeFullLogContext}
              settings={settings}
              useRag={useCondenserRag}
              ragDocs={condenserRagDocs}
            />
          </div>
        </TabsContent>

        <TabsContent value="agent" className="mt-0">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Input chatlog (shared)</Label>
              <Textarea
                className="min-h-[120px] font-mono text-xs"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="**User:** ...&#10;&#10;**Assistant:** ..."
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={loadActiveConversation}
                disabled={!activeConv}
              >
                <ScrollText className="mr-2 h-4 w-4" />
                Load active chat
              </Button>
              <Button type="button" variant="outline" size="sm" asChild>
                <label className="cursor-pointer">
                  <FileUp className="mr-2 h-4 w-4" />
                  Upload .txt / .md
                  <input
                    type="file"
                    accept=".txt,.md,.markdown"
                    className="hidden"
                    onChange={handleFile}
                  />
                </label>
              </Button>
            </div>
            <ChatlogCondenserAgentPanel
              apiUrl={apiUrl}
              apiReady={apiReady}
              inputText={inputText}
              modelName={modelName}
              setModelName={setModelName}
              modelOptions={modelOptions}
              includeFullLogContext={includeFullLogContext}
              useRag={useCondenserRag}
              ragDocs={condenserRagDocs}
              onRestoreOriginalLog={(log) => {
                if (log && String(log).trim()) setInputText(log);
              }}
            />
          </div>
        </TabsContent>

        <TabsContent value="quick" className="mt-0 space-y-4">
      <p className="text-sm text-muted-foreground leading-relaxed">
        Produces a dense draft that preserves every reasoning move, correction, and thread-shift —
        not bullet takeaways. Use the output with an orienting prompt when sharing a long exchange
        with another model. Structural fidelity beats hitting a target ratio.
      </p>

      {!apiReady && (
        <Alert>
          <AlertTitle>Waiting for backend</AlertTitle>
          <AlertDescription>Start Mirid and wait for storage hydration.</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-2">
          <Label>Model (long-context API recommended)</Label>
          <Select value={modelName} onValueChange={setModelName}>
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
        <div className="space-y-2">
          <Label>Soft target ratio ({Math.round(targetRatio * 100)}%)</Label>
          <Slider
            min={1}
            max={100}
            step={1}
            value={[Math.min(100, Math.max(1, Math.round(targetRatio * 100)))]}
            onValueChange={([v]) => setTargetRatio(v / 100)}
          />
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-2">
          <Label>Chunk budget (tokens)</Label>
          <Input
            type="number"
            min={1}
            value={chunkTokens}
            onChange={(e) => {
              const n = Number(e.target.value);
              setChunkTokens(Number.isFinite(n) ? n : CHUNK_TOKENS_DEFAULT);
            }}
            onBlur={() =>
              setChunkTokens((v) => {
                const n = Math.round(Number(v));
                return Number.isFinite(n) && n >= 1 ? n : CHUNK_TOKENS_DEFAULT;
              })
            }
          />
          <p className="text-xs text-muted-foreground leading-relaxed">
            Segment size for each processing pass (turn packing). Smaller budget → more chunks and
            more LLM calls; not a fixed count of 2. Total chunks = transcript length ÷ this budget
            (after parse).
          </p>
        </div>
        <div className="space-y-2">
          <Label>Overlap turns between chunks</Label>
          <Input
            type="number"
            min={0}
            value={overlapTurns}
            onChange={(e) => {
              const n = Number(e.target.value);
              setOverlapTurns(Number.isFinite(n) ? n : 5);
            }}
            onBlur={() =>
              setOverlapTurns((v) => {
                const n = Math.round(Number(v));
                return Number.isFinite(n) && n >= 0 ? n : 5;
              })
            }
          />
          <p className="text-xs text-muted-foreground leading-relaxed">
            How many speaker turns repeat at each chunk boundary for continuity — not how many
            chunks to run. Default 5; set 0 for no overlap.
          </p>
        </div>
      </div>

      {chunkPlan && (
        <p className="text-sm text-muted-foreground">
          {chunkPlan.isExact ? 'Estimated' : 'Approx.'}{' '}
          <span className="font-medium text-foreground">
            {chunkPlan.chunkCount} chunk{chunkPlan.chunkCount === 1 ? '' : 's'}
          </span>{' '}
          ({chunkPlan.llmPasses} LLM pass{chunkPlan.llmPasses === 1 ? '' : 'es'}
          {runEval ? ', includes eval' : ''}
          {chunkPlan.chunkCount > 1 ? ', includes stitch' : ''}) —{' '}
          {chunkPlan.isExact
            ? 'from parsed turns'
            : `~${chunkPlan.tokensEst.toLocaleString()} tokens; use Preview parse for exact count`}
        </p>
      )}

      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <Checkbox
            id="full-log-context"
            checked={includeFullLogContext}
            onCheckedChange={(c) => setIncludeFullLogContext(!!c)}
          />
          <Label htmlFor="full-log-context" className="font-normal cursor-pointer">
            Include full chatlog on every LLM pass (recommended for intellectual threads)
          </Label>
        </div>
        <div className="flex items-center gap-2">
          <Checkbox id="run-eval" checked={runEval} onCheckedChange={(c) => setRunEval(!!c)} />
          <Label htmlFor="run-eval" className="font-normal cursor-pointer">
            Run reconstruction-fidelity eval after condensing
          </Label>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="outline" size="sm" onClick={loadActiveConversation} disabled={!activeConv}>
          <ScrollText className="mr-2 h-4 w-4" />
          Load active chat
        </Button>
        <Button type="button" variant="outline" size="sm" asChild>
          <label className="cursor-pointer">
            <FileUp className="mr-2 h-4 w-4" />
            Upload .txt / .md
            <input type="file" accept=".txt,.md,.markdown" className="hidden" onChange={handleFile} />
          </label>
        </Button>
        <Button type="button" variant="outline" size="sm" onClick={handleParsePreview} disabled={!inputText.trim()}>
          Preview parse
        </Button>
        <Button type="button" onClick={handleCondense} disabled={busy || !inputText.trim()}>
          {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
          Condense
        </Button>
      </div>

      {parsePreview && (
        <p className="text-xs text-muted-foreground">
          Parsed {parsePreview.turn_count} turns ({parsePreview.tokens_est} tokens est.), speakers:{' '}
          {(parsePreview.speakers || []).join(', ')}
          {typeof parsePreview.estimated_chunk_count === 'number' && (
            <>
              {' '}
              → {parsePreview.estimated_chunk_count} chunk
              {parsePreview.estimated_chunk_count === 1 ? '' : 's'},{' '}
              {parsePreview.estimated_llm_passes} LLM passes
            </>
          )}
        </p>
      )}

      <div className="space-y-2">
        <Label>Input chatlog</Label>
        <Textarea
          className="min-h-[200px] font-mono text-xs"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="**User:** ...&#10;&#10;**Assistant:** ..."
        />
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {result && (
        <div className="space-y-3 rounded-xl border border-border/80 bg-card/60 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">
              {result.stats?.input_tokens_est} → {result.stats?.output_tokens_est} tokens (
              {Math.round((result.stats?.achieved_ratio || 0) * 100)}%)
            </span>
            <span className="text-sm font-medium text-primary">
              {result.stats?.chunk_count} chunk{result.stats?.chunk_count === 1 ? '' : 's'} processed
            </span>
            {typeof fidelity === 'number' && (
              <span className="text-sm text-muted-foreground">
                Eval fidelity: {(fidelity * 100).toFixed(0)}%
              </span>
            )}
            <Button type="button" variant="outline" size="sm" onClick={copyOutput}>
              <Copy className="mr-2 h-4 w-4" />
              Copy draft
            </Button>
          </div>

          {result.stats?.context_warning && (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              {result.stats.context_warning}
              {result.stats?.context_tokens_est
                ? ` (peak per-call context ~${result.stats.context_tokens_est.toLocaleString()} tokens est.)`
                : ''}
            </p>
          )}

          {result.eval?.summary?.failure_modes?.length > 0 && (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              Failure modes: {result.eval.summary.failure_modes.join(', ')}
            </p>
          )}

          <Textarea
            className="min-h-[280px] font-mono text-xs"
            readOnly
            value={result.condensed_markdown || ''}
          />
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        API: <code className="px-1 rounded bg-muted">POST /memory/chatlog-condenser/condense</code> — two-stage
        skeleton then dense draft (full log + segment per pass); multi-chunk stitch when needed.
      </p>
        </TabsContent>
      </Tabs>
    </div>
  );
}
