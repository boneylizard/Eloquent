import React, { useCallback, useEffect, useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { fetchWithTimeout } from '../config/api';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { Loader2, FolderSearch, Search, Trash2, Database } from 'lucide-react';

const STORAGE_FOLDER_KEY = 'eloquent:transcriptCorpusFolder';
const STORAGE_CORPUS_KEY = 'eloquent:transcriptCorpusId';

function loadStored(key, fallback = '') {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}

function saveStored(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

function scoreLabel(score) {
  if (score >= 0.55) return 'Strong match';
  if (score >= 0.35) return 'Good match';
  return 'Possible match';
}

function formatApiError(data, statusText) {
  const d = data?.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d.map((x) => (typeof x === 'object' ? x.msg || JSON.stringify(x) : String(x))).join('; ');
  }
  if (d && typeof d === 'object') return JSON.stringify(d);
  return statusText || 'Request failed';
}

function phaseLabel(job) {
  if (!job) return '';
  const p = job.phase || job.status;
  const labels = {
    queued: 'Queued',
    starting: 'Starting',
    scanning: 'Scanning folder',
    chunking: 'Chunking files',
    loading_model: 'Loading embedding model',
    embedding: 'Embedding chunks',
    saving: 'Saving index',
    done: 'Complete',
    completed: 'Complete',
    failed: 'Failed',
  };
  return labels[p] || p || 'Working';
}

export default function TranscriptCorpusPanel() {
  const { PRIMARY_API_URL, portsReady, storageHydrated } = useApp();
  const apiReady = portsReady && storageHydrated;
  const base = PRIMARY_API_URL;

  const [status, setStatus] = useState(null);
  const [corpora, setCorpora] = useState([]);
  const [selectedCorpusId, setSelectedCorpusId] = useState(() => loadStored(STORAGE_CORPUS_KEY));
  const [folderPath, setFolderPath] = useState(() => loadStored(STORAGE_FOLDER_KEY));
  const [corpusName, setCorpusName] = useState('');
  const [recursive, setRecursive] = useState(true);

  const [indexing, setIndexing] = useState(false);
  const [indexJob, setIndexJob] = useState(null);
  const [indexError, setIndexError] = useState('');

  const [query, setQuery] = useState('');
  const [keyword, setKeyword] = useState('');
  const [fileFilter, setFileFilter] = useState('');
  const [minFirstPerson, setMinFirstPerson] = useState(0);
  const [useFirstPersonFilter, setUseFirstPersonFilter] = useState(false);
  const [pageSize, setPageSize] = useState(25);
  const [minScore, setMinScore] = useState(0.2);
  const [loadingMore, setLoadingMore] = useState(false);

  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [activityLog, setActivityLog] = useState([]);

  const pushLog = useCallback((msg) => {
    const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
    console.info('[transcript-corpus]', msg);
    setActivityLog((prev) => [line, ...prev].slice(0, 40));
  }, []);

  const refreshStatus = useCallback(async () => {
    if (!apiReady) return;
    try {
      const res = await fetchWithTimeout(`${base}/corpus/status`);
      if (res.ok) setStatus(await res.json());
    } catch (e) {
      console.warn('corpus status', e);
    }
  }, [apiReady, base]);

  const refreshCorpora = useCallback(async () => {
    if (!apiReady) return;
    try {
      const res = await fetchWithTimeout(`${base}/corpus/list`);
      if (!res.ok) return;
      const data = await res.json();
      const list = data.corpora || [];
      setCorpora(list);
      if (list.length && !list.some((c) => c.id === selectedCorpusId)) {
        const next = list[0].id;
        setSelectedCorpusId(next);
        saveStored(STORAGE_CORPUS_KEY, next);
      }
    } catch (e) {
      console.warn('corpus list', e);
    }
  }, [apiReady, base, selectedCorpusId]);

  useEffect(() => {
    refreshStatus();
    refreshCorpora();
  }, [refreshStatus, refreshCorpora]);

  useEffect(() => {
    const jobId = indexJob?.job_id;
    if (!jobId || indexJob.status === 'completed' || indexJob.status === 'failed') {
      return undefined;
    }

    const poll = async () => {
      try {
        const res = await fetchWithTimeout(`${base}/corpus/job/${jobId}`, {}, 15000);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          pushLog(`Poll error ${res.status}: ${formatApiError(data, res.statusText)}`);
          return;
        }
        setIndexJob((prev) => {
          const changed =
            prev?.phase !== data.phase ||
            prev?.files_done !== data.files_done ||
            prev?.embed_done !== data.embed_done ||
            prev?.status !== data.status;
          if (changed && data.message) {
            pushLog(data.message);
          } else if (changed) {
            pushLog(`${phaseLabel(data)} — ${data.current_file || data.status}`);
          }
          return data;
        });
        if (data.status === 'completed') {
          setIndexing(false);
          pushLog(`Index complete: ${data.chunk_count ?? '?'} chunks`);
          await refreshCorpora();
          if (data.corpus_id) {
            setSelectedCorpusId(data.corpus_id);
            saveStored(STORAGE_CORPUS_KEY, data.corpus_id);
          }
        }
        if (data.status === 'failed') {
          setIndexing(false);
          const err = data.error || 'Indexing failed';
          setIndexError(err);
          pushLog(`Failed: ${err}`);
        }
      } catch (e) {
        pushLog(`Poll failed: ${e.message || e}`);
      }
    };

    poll();
    const id = setInterval(poll, 1000);
    return () => clearInterval(id);
  }, [indexJob?.job_id, indexJob?.status, base, refreshCorpora, pushLog]);

  const handleIndex = async () => {
    setIndexError('');
    const path = folderPath.trim();
    if (!path) {
      setIndexError('Enter the folder path containing your .txt transcripts.');
      return;
    }
    if (!status?.vector_search_available) {
      setIndexError('Vector search is not available on the backend. Install sentence-transformers and faiss-cpu, then restart.');
      pushLog('Index blocked: vector search unavailable');
      return;
    }
    saveStored(STORAGE_FOLDER_KEY, path);
    setIndexing(true);
    setIndexJob(null);
    pushLog(`Sending index request to ${base}/corpus/index …`);
    pushLog(`Folder: ${path}`);
    try {
      const res = await fetchWithTimeout(
        `${base}/corpus/index`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            folder_path: path,
            corpus_name: corpusName.trim() || undefined,
            recursive,
            background: true,
          }),
        },
        120000
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(formatApiError(data, res.statusText));
      }
      if (data.job_id) {
        pushLog(`Job started: ${data.job_id} (polling every second)`);
        setIndexJob({ job_id: data.job_id, status: 'queued', phase: 'queued', message: 'Job accepted by server' });
      } else if (data.id) {
        setIndexing(false);
        pushLog(`Indexed synchronously: ${data.chunk_count} chunks`);
        await refreshCorpora();
        setSelectedCorpusId(data.id);
        saveStored(STORAGE_CORPUS_KEY, data.id);
      }
    } catch (e) {
      setIndexing(false);
      const msg = e.message || String(e);
      setIndexError(msg);
      pushLog(`Index request error: ${msg}`);
    }
  };

  const runSearch = async ({ offset = 0, append = false } = {}) => {
    const q = query.trim();
    if (!selectedCorpusId) {
      setSearchError('Index a folder first, or select a corpus.');
      return;
    }
    if (!q) {
      setSearchError('Enter a search query — themes work better than single keywords.');
      return;
    }
    if (append) {
      setLoadingMore(true);
    } else {
      setSearchError('');
      setSearchResult(null);
      setSearching(true);
    }
    try {
      const body = {
        corpus_id: selectedCorpusId,
        query: q,
        top_k: pageSize,
        offset,
        min_score: minScore,
        source_file_contains: fileFilter.trim() || undefined,
        keyword: keyword.trim() || undefined,
      };
      if (useFirstPersonFilter && minFirstPerson > 0) {
        body.min_first_person = minFirstPerson / 100;
      }
      const res = await fetchWithTimeout(
        `${base}/corpus/search`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
        120000
      );
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data, res.statusText));
      if (append && searchResult?.results) {
        setSearchResult({
          ...data,
          results: [...searchResult.results, ...(data.results || [])],
        });
      } else {
        setSearchResult(data);
      }
      pushLog(
        append
          ? `Loaded more: ${data.result_count} (total shown ${(append ? searchResult?.results?.length || 0 : 0) + data.result_count} / ${data.total_matches} matches)`
          : `Search: ${data.total_matches} matches (showing ${data.result_count})`
      );
    } catch (e) {
      setSearchError(e.message || String(e));
    } finally {
      setSearching(false);
      setLoadingMore(false);
    }
  };

  const handleSearch = () => runSearch({ offset: 0, append: false });

  const handleLoadMore = () => {
    if (!searchResult?.has_more) return;
    const next = searchResult.next_offset ?? searchResult.results?.length ?? 0;
    runSearch({ offset: next, append: true });
  };

  const handleDeleteCorpus = async () => {
    if (!selectedCorpusId) return;
    if (!window.confirm('Delete this corpus index from disk?')) return;
    try {
      const res = await fetchWithTimeout(`${base}/corpus/${selectedCorpusId}`, { method: 'DELETE' });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Delete failed');
      }
      setSelectedCorpusId('');
      setSearchResult(null);
      await refreshCorpora();
    } catch (e) {
      setSearchError(e.message || String(e));
    }
  };

  const selectedMeta = corpora.find((c) => c.id === selectedCorpusId);

  return (
    <div className="transcript-corpus space-y-8 max-w-4xl mx-auto pb-16">
      <header className="rounded-xl border border-border bg-card/80 p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <FolderSearch className="h-8 w-8 text-primary shrink-0 mt-0.5" />
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Transcript search</h1>
            <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
              Point at a folder of <strong>.txt</strong> files, index once, then search by{' '}
              <em>meaning</em> on this tab only. This is separate from chat <strong>Web Search</strong>{' '}
              (globe in the chat bar).
            </p>
          </div>
        </div>
        {status && !status.vector_search_available && (
          <Alert variant="destructive" className="mt-4">
            <AlertTitle>Vector search unavailable</AlertTitle>
            <AlertDescription>
              Install backend dependencies:{' '}
              <code className="text-xs">pip install sentence-transformers faiss-cpu</code>
              then restart the backend. Indexing will not run until this is fixed.
            </AlertDescription>
          </Alert>
        )}
        {apiReady && base && (
          <p className="text-xs text-muted-foreground mt-3 font-mono">API: {base}</p>
        )}
      </header>

      <section className="rounded-xl border border-border/80 bg-card/60 p-5 space-y-4">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Database className="h-5 w-5" /> 1 · Index a folder
        </h2>
        <div className="space-y-2">
          <Label htmlFor="corpus-folder">Folder path</Label>
          <Input
            id="corpus-folder"
            placeholder="C:\Users\you\transcripts"
            value={folderPath}
            onChange={(e) => setFolderPath(e.target.value)}
            disabled={indexing}
          />
          <p className="text-xs text-muted-foreground">
            All <code>.txt</code> files in this folder{recursive ? ' (and subfolders)' : ''} will be
            chunked and embedded locally.
          </p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="corpus-name">Display name (optional)</Label>
            <Input
              id="corpus-name"
              placeholder="e.g. March batch transcripts"
              value={corpusName}
              onChange={(e) => setCorpusName(e.target.value)}
              disabled={indexing}
            />
          </div>
          <div className="flex items-end gap-2 pb-1">
            <Switch id="corpus-recursive" checked={recursive} onCheckedChange={setRecursive} />
            <Label htmlFor="corpus-recursive" className="cursor-pointer">
              Include subfolders
            </Label>
          </div>
        </div>
        {indexError && (
          <Alert variant="destructive">
            <AlertDescription>{indexError}</AlertDescription>
          </Alert>
        )}
        {(indexing || indexJob) && (
          <div className="rounded-lg border border-primary/30 bg-primary/5 p-4 space-y-3 text-sm">
            <div className="font-medium text-foreground">
              {indexing ? (
                <span className="inline-flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {phaseLabel(indexJob)}
                </span>
              ) : (
                phaseLabel(indexJob)
              )}
            </div>
            {indexJob?.message && (
              <p className="text-muted-foreground leading-relaxed">{indexJob.message}</p>
            )}
            {indexJob?.files_total != null && indexJob.files_total > 0 && (
              <p className="text-muted-foreground tabular-nums">
                Files: {indexJob.files_done ?? 0} / {indexJob.files_total}
                {indexJob.current_file ? ` · ${indexJob.current_file}` : ''}
              </p>
            )}
            {indexJob?.embed_total != null && indexJob.embed_total > 0 && (
              <p className="text-muted-foreground tabular-nums">
                Embeddings: {indexJob.embed_done ?? 0} / {indexJob.embed_total}
              </p>
            )}
            {indexJob?.chunks_indexed != null && (
              <p className="text-muted-foreground tabular-nums">
                Chunks so far: {indexJob.chunks_indexed}
              </p>
            )}
            {indexJob?.job_id && (
              <p className="text-xs text-muted-foreground font-mono">Job {indexJob.job_id}</p>
            )}
          </div>
        )}
        {indexJob?.status === 'completed' && !indexing && (
          <p className="text-sm text-green-600 dark:text-green-400">
            Indexed {indexJob.chunk_count} chunks from {indexJob.file_count} files.
          </p>
        )}
        {activityLog.length > 0 && (
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Activity log</Label>
            <pre className="max-h-40 overflow-y-auto rounded-md border border-border/60 bg-muted/30 p-3 text-xs font-mono leading-relaxed whitespace-pre-wrap">
              {activityLog.join('\n')}
            </pre>
          </div>
        )}
        <Button type="button" onClick={handleIndex} disabled={!apiReady || indexing}>
          {indexing ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
          {indexing ? 'Indexing…' : 'Index folder'}
        </Button>
      </section>

      <section className="rounded-xl border border-border/80 bg-card/60 p-5 space-y-4">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Search className="h-5 w-5" /> 2 · Search
        </h2>
        <div className="space-y-2">
          <Label>Corpus</Label>
          <div className="flex flex-wrap gap-2">
            <Select
              value={selectedCorpusId || undefined}
              onValueChange={(v) => {
                setSelectedCorpusId(v);
                saveStored(STORAGE_CORPUS_KEY, v);
              }}
            >
              <SelectTrigger className="w-full sm:w-72">
                <SelectValue placeholder="Select indexed corpus" />
              </SelectTrigger>
              <SelectContent>
                {corpora.map((c) => (
                  <SelectItem key={c.id} value={c.id}>
                    {c.name} ({c.chunk_count} chunks)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedCorpusId && (
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={handleDeleteCorpus}
                title="Delete corpus"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </div>
          {selectedMeta && (
            <p className="text-xs text-muted-foreground">
              {selectedMeta.file_count} files · indexed {selectedMeta.indexed_at?.slice(0, 10)} ·{' '}
              {selectedMeta.source_folder}
            </p>
          )}
        </div>
        <div className="space-y-2">
          <Label htmlFor="corpus-query">Semantic query</Label>
          <Textarea
            id="corpus-query"
            rows={3}
            placeholder="e.g. moments of shame or embarrassment, childhood memories with parents, feeling trapped at work"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="corpus-keyword">Must contain word (optional)</Label>
            <Input
              id="corpus-keyword"
              placeholder="exact substring"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="corpus-file-filter">Filename contains (optional)</Label>
            <Input
              id="corpus-file-filter"
              placeholder="e.g. session_03"
              value={fileFilter}
              onChange={(e) => setFileFilter(e.target.value)}
            />
          </div>
        </div>
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Switch
              id="fp-filter"
              checked={useFirstPersonFilter}
              onCheckedChange={setUseFirstPersonFilter}
            />
            <Label htmlFor="fp-filter">First-person filter (I, me, my, we…)</Label>
          </div>
          {useFirstPersonFilter && (
            <div className="space-y-2 pl-1">
              <Label>Minimum first-person density: {(minFirstPerson / 100).toFixed(2)}</Label>
              <Slider
                value={[minFirstPerson]}
                min={0}
                max={8}
                step={1}
                onValueChange={([v]) => setMinFirstPerson(v)}
              />
              <p className="text-xs text-muted-foreground">
                Higher = more &quot;I/me/my&quot; language in the passage (narrative voice vs reported speech).
              </p>
            </div>
          )}
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label>Match strictness: {minScore.toFixed(2)}</Label>
            <Slider
              value={[minScore * 100]}
              min={5}
              max={50}
              step={1}
              onValueChange={([v]) => setMinScore(v / 100)}
            />
            <p className="text-xs text-muted-foreground">
              Higher = fewer, stronger matches. Count varies by query — not a fixed 25.
            </p>
          </div>
          <div className="space-y-2">
            <Label>Results per page</Label>
            <Select value={String(pageSize)} onValueChange={(v) => setPageSize(Number(v))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="10">10</SelectItem>
                <SelectItem value="25">25</SelectItem>
                <SelectItem value="50">50</SelectItem>
                <SelectItem value="100">100</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        {searchError && (
          <Alert variant="destructive">
            <AlertDescription>{searchError}</AlertDescription>
          </Alert>
        )}
        <Button type="button" onClick={handleSearch} disabled={!apiReady || searching}>
          {searching ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
          Search corpus
        </Button>
      </section>

      {searchResult && (
        <section className="space-y-4">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold">
              {searchResult.total_matches != null ? (
                <>
                  Showing {searchResult.results?.length ?? 0} of {searchResult.total_matches} matches
                </>
              ) : (
                <>Results ({searchResult.result_count})</>
              )}
            </h2>
            {searchResult.total_matches != null && searchResult.total_matches > 0 && (
              <p className="text-sm text-muted-foreground">
                Best match score {searchResult.score_range?.best_overall?.toFixed(3) ?? '—'}
                {searchResult.score_range?.page_min != null && (
                  <>
                    {' '}
                    · this page {searchResult.score_range.page_max?.toFixed(3)}–
                    {searchResult.score_range.page_min?.toFixed(3)}
                  </>
                )}
                {searchResult.total_matches <= (searchResult.results?.length ?? 0) ? (
                  <span> · all matches shown</span>
                ) : null}
              </p>
            )}
          </div>
          {searchResult.total_matches === 0 && (
            <p className="text-sm text-muted-foreground">
              No matches above your strictness threshold. Try a looser query or lower match strictness.
            </p>
          )}
          <ul className="space-y-4">
            {searchResult.results?.map((r) => (
              <li
                key={r.chunk_id}
                className="rounded-lg border border-border/70 bg-muted/20 p-4 space-y-2"
              >
                <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">{scoreLabel(r.score)}</span>
                  <span>· score {r.score}</span>
                  <span>· first-person {(r.first_person_ratio * 100).toFixed(1)}%</span>
                  <span className="truncate max-w-full" title={r.source_path}>
                    · {r.source_file}
                  </span>
                </div>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{r.text}</p>
              </li>
            ))}
          </ul>
          {searchResult.has_more && (
            <Button
              type="button"
              variant="outline"
              onClick={handleLoadMore}
              disabled={loadingMore || searching}
            >
              {loadingMore ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Load more ({(searchResult.total_matches ?? 0) - (searchResult.results?.length ?? 0)} remaining)
            </Button>
          )}
        </section>
      )}
    </div>
  );
}
