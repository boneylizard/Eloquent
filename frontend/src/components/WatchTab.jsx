import React, { useCallback, useRef, useState } from 'react';
import { useVideoWatch } from '../contexts/VideoWatchContext';
import { parsePlaylistText } from '../utils/watchPlaylistParse';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { ScrollArea } from './ui/scroll-area';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import {
  Clapperboard,
  Trash2,
  ChevronUp,
  ChevronDown,
  Maximize2,
  PictureInPicture2,
  Link2,
  Loader2,
} from 'lucide-react';
import DidPipelineOverlay from './DidPipelineOverlay';

export default function WatchTab() {
  const {
    items,
    currentIndex,
    current,
    registerWatchHost,
    goNext,
    goPrev,
    addItem,
    removeItem,
    replacePlaylist,
    playIndex,
    requestFullscreen,
    requestPip,
    dockMini,
    setDockMini,
    muted,
    setMuted,
  } = useVideoWatch();

  const videoMountRef = useCallback(
    (node) => {
      registerWatchHost(node);
    },
    [registerWatchHost]
  );

  const [urlInput, setUrlInput] = useState('');
  const [titleInput, setTitleInput] = useState('');
  const [fetchUrl, setFetchUrl] = useState('');
  const [pasteText, setPasteText] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef(null);
  const pipSupported = typeof document !== 'undefined' && document.pictureInPictureEnabled;

  const [showDidPipeline, setShowDidPipeline] = useState(false);

  const handleFetchPlaylist = async () => {
    const u = fetchUrl.trim();
    if (!u) {
      setError('Enter a playlist URL (http/https on your Tailscale or LAN server).');
      return;
    }
    setBusy(true);
    setError('');
    try {
      const res = await fetch(u, { method: 'GET' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const parsed = parsePlaylistText(text);
      if (!parsed.length) throw new Error('No playable URLs found (JSON array, M3U, or one URL per line).');
      replacePlaylist(parsed, 0);
    } catch (e) {
      setError(
        e.message ||
          'Could not load playlist. CORS: the file server must allow this origin, or host the playlist JSON on the same host as Eloquent.'
      );
    } finally {
      setBusy(false);
    }
  };

  const handlePasteApply = () => {
    setError('');
    const parsed = parsePlaylistText(pasteText);
    if (!parsed.length) {
      setError('Paste JSON, M3U, or plain URLs (one per line).');
      return;
    }
    replacePlaylist(parsed, 0);
  };

  const handleFile = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setError('');
    try {
      const text = await f.text();
      const parsed = parsePlaylistText(text);
      if (!parsed.length) {
        setError('No URLs found in file.');
        return;
      }
      replacePlaylist(parsed, 0);
    } catch (err) {
      setError(String(err.message || err));
    }
    e.target.value = '';
  };

  return (
    <div className="mx-auto max-w-6xl space-y-4 pb-8">
      <div className="flex flex-wrap items-center gap-2">
        <Clapperboard className="h-7 w-7 text-primary" aria-hidden />
        <h1 className="text-2xl font-bold tracking-tight">Watch</h1>
        <span className="text-sm text-muted-foreground">
          Playlist + fullscreen. Use Chat with mini player or Picture-in-Picture.
        </span>
        <Button type="button" variant="secondary" size="sm" className="ml-auto" onClick={() => setShowDidPipeline(true)}>
          D-ID batch pipeline
        </Button>
      </div>

      <DidPipelineOverlay open={showDidPipeline} onClose={() => setShowDidPipeline(false)} />

      <Alert>
        <AlertTitle>Formats</AlertTitle>
        <AlertDescription>
          The browser plays whatever it supports natively (usually <strong>MP4 H.264</strong>,{' '}
          <strong>WebM</strong>, <strong>Ogg</strong>). Other containers (MKV, AVI) may not play without
          transcoding. Use direct file URLs your TV browser can reach (Tailscale IP, same LAN, or same
          origin as this app).
        </AlertDescription>
      </Alert>

      {error ? (
        <Alert variant="destructive">
          <AlertTitle>Playlist</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-[minmax(260px,320px)_1fr]">
        <div className="space-y-3 rounded-xl border border-border bg-card/60 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Playlist</p>
          <ScrollArea className="h-[min(40vh,320px)] pr-3">
            <ul className="space-y-1">
              {items.length === 0 ? (
                <li className="text-sm text-muted-foreground">Add URLs or load a playlist file.</li>
              ) : (
                items.map((it, idx) => (
                  <li key={it.id}>
                    <div
                      role="button"
                      tabIndex={0}
                      onClick={() => playIndex(idx)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          playIndex(idx);
                        }
                      }}
                      className={`flex w-full cursor-pointer items-start gap-2 rounded-md border px-2 py-2 text-left text-sm transition-colors ${
                        idx === currentIndex
                          ? 'border-primary bg-primary/10 text-foreground'
                          : 'border-transparent hover:bg-muted/60'
                      }`}
                    >
                      <span className="truncate flex-1">{it.title}</span>
                      <button
                        type="button"
                        className="shrink-0 rounded p-1 text-muted-foreground hover:bg-destructive/15 hover:text-destructive"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeItem(it.id);
                        }}
                        aria-label="Remove"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </li>
                ))
              )}
            </ul>
          </ScrollArea>

          <div className="space-y-2 border-t border-border pt-3">
            <Label className="text-xs">Add one URL</Label>
            <Input
              placeholder="https://your-pc:8080/video.mp4"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
            />
            <Input placeholder="Title (optional)" value={titleInput} onChange={(e) => setTitleInput(e.target.value)} />
            <Button
              type="button"
              variant="secondary"
              className="w-full"
              onClick={() => {
                addItem(urlInput, titleInput);
                setUrlInput('');
                setTitleInput('');
              }}
            >
              Add to playlist
            </Button>
          </div>

          <div className="space-y-2 border-t border-border pt-3">
            <Label className="text-xs flex items-center gap-1">
              <Link2 className="h-3 w-3" /> Load playlist from network URL
            </Label>
            <Input
              placeholder="https://100.x.x.x:9000/playlist.json"
              value={fetchUrl}
              onChange={(e) => setFetchUrl(e.target.value)}
            />
            <Button type="button" className="w-full" disabled={busy} onClick={handleFetchPlaylist}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              {busy ? ' Loading…' : 'Fetch playlist'}
            </Button>
          </div>

          <div className="space-y-2 border-t border-border pt-3">
            <Label className="text-xs">Paste playlist (JSON / M3U / URLs)</Label>
            <textarea
              className="flex min-h-[100px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              placeholder='["https://.../a.mp4","https://.../b.mp4"]'
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
            />
            <Button type="button" variant="outline" className="w-full" onClick={handlePasteApply}>
              Replace playlist from paste
            </Button>
          </div>

          <div className="space-y-2 border-t border-border pt-3">
            <input ref={fileRef} type="file" accept=".m3u,.m3u8,.json,.txt" className="hidden" onChange={handleFile} />
            <Button type="button" variant="outline" className="w-full" onClick={() => fileRef.current?.click()}>
              Load playlist file…
            </Button>
          </div>

          <div className="flex items-center justify-between gap-2 border-t border-border pt-3">
            <div>
              <Label htmlFor="dock-mini" className="text-sm font-medium">
                Mini player while in Chat
              </Label>
              <p className="text-xs text-muted-foreground">Corner player when you leave this tab.</p>
            </div>
            <Switch id="dock-mini" checked={dockMini} onCheckedChange={setDockMini} />
          </div>

          <div className="flex items-center justify-between gap-2 border-t border-border pt-3">
            <div>
              <Label htmlFor="video-muted" className="text-sm font-medium">
                Mute video audio
              </Label>
              <p className="text-xs text-muted-foreground">
                Keep this on while talking to AI with mic/pedals.
              </p>
            </div>
            <Switch id="video-muted" checked={muted} onCheckedChange={setMuted} />
          </div>
        </div>

        <div className="flex min-h-[min(55vh,480px)] flex-col gap-3 rounded-xl border border-border bg-card/40 p-3">
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="outline" size="sm" onClick={goPrev} disabled={!items.length}>
              <ChevronUp className="h-4 w-4 rotate-[-90deg]" /> Prev
            </Button>
            <Button type="button" variant="outline" size="sm" onClick={goNext} disabled={!items.length}>
              Next <ChevronDown className="h-4 w-4 rotate-[-90deg]" />
            </Button>
            <Button type="button" variant="outline" size="sm" onClick={requestFullscreen} disabled={!current}>
              <Maximize2 className="h-4 w-4" /> Fullscreen
            </Button>
            {pipSupported ? (
              <Button type="button" variant="outline" size="sm" onClick={requestPip} disabled={!current}>
                <PictureInPicture2 className="h-4 w-4" /> PiP
              </Button>
            ) : null}
            <Button
              type="button"
              variant={muted ? 'default' : 'outline'}
              size="sm"
              onClick={() => setMuted((m) => !m)}
              disabled={!current}
            >
              {muted ? 'Unmute' : 'Mute'}
            </Button>
          </div>
          <div
            ref={videoMountRef}
            className="relative flex flex-1 min-h-[220px] items-center justify-center overflow-hidden rounded-lg bg-black"
          />
          {current ? (
            <p className="text-center text-xs text-muted-foreground truncate px-2" title={current.url}>
              {current.title}
            </p>
          ) : (
            <p className="text-center text-sm text-muted-foreground">Add items to start.</p>
          )}
        </div>
      </div>
    </div>
  );
}
