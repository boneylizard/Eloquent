import React, { useState, useEffect, useCallback, useMemo } from 'react';
import * as indexedDbStorage from '../utils/indexedDbStorage';
import {
  emergencyRecoverAllConversations,
  recoverChatsFromShards,
  repairAndPurgeGhostChats,
} from '../utils/conversationStorage';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Checkbox } from './ui/checkbox';
import { RefreshCw, Trash2, Loader2 } from 'lucide-react';
import { Label } from './ui/label';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem
} from './ui/select';

function formatBytes(n) {
  if (n == null || Number.isNaN(n)) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

const KIND_LABEL = {
  'conversation-shard': 'Conversation messages',
  'message-variants': 'Message variants',
  'conversations-index': 'Conversation list / index',
  characters: 'Characters',
  profiles: 'User profiles',
  settings: 'Settings',
  memory: 'Memory / story',
  other: 'Other'
};

function rowId(storage, key) {
  return `${storage}\x1f${key}`;
}

function parseRowId(id) {
  const i = id.indexOf('\x1f');
  return { storage: id.slice(0, i), key: id.slice(i + 1) };
}

/**
 * Browse and delete browser storage (IndexedDB + stray localStorage) to recover quota.
 */
export default function LocalStorageManager({ conversations = [] }) {
  const [idbRows, setIdbRows] = useState([]);
  const [lsRows, setLsRows] = useState([]);
  const [quota, setQuota] = useState(null);
  const [loading, setLoading] = useState(true);
  const [filterText, setFilterText] = useState('');
  const [kindFilter, setKindFilter] = useState('all');
  const [deleting, setDeleting] = useState(false);
  const [statusNote, setStatusNote] = useState('');
  const [selected, setSelected] = useState(() => new Set());

  const notifyStorageChanged = (key) => {
    try {
      window.dispatchEvent(new CustomEvent('eloquent-storage-changed', { detail: { key } }));
    } catch (_) { /* noop */ }
  };

  const convNameById = useMemo(() => {
    const m = new Map();
    (conversations || []).forEach((c) => {
      if (c?.id) m.set(c.id, c.name || c.title || c.id);
    });
    return m;
  }, [conversations]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [inv, ls, q] = await Promise.all([
        indexedDbStorage.getStorageInventory(),
        Promise.resolve(indexedDbStorage.listNonIdbLocalStorageKeys()),
        indexedDbStorage.getStorageQuotaInfo()
      ]);
      setIdbRows(inv);
      setLsRows(ls);
      setQuota(q);
    } catch (e) {
      console.warn('[LocalStorageManager] load failed', e);
      setIdbRows([]);
      setLsRows([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const filteredIdb = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    return idbRows.filter((row) => {
      if (kindFilter !== 'all' && row.kind !== kindFilter) return false;
      if (!q) return true;
      if (row.key.toLowerCase().includes(q)) return true;
      if (row.convId && row.convId.toLowerCase().includes(q)) return true;
      return false;
    });
  }, [idbRows, filterText, kindFilter]);

  const filteredLs = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    return lsRows.filter((row) => {
      if (!q) return true;
      return row.key.toLowerCase().includes(q);
    });
  }, [lsRows, filterText]);

  const totalIndexedBytes = useMemo(
    () => idbRows.reduce((s, r) => s + r.sizeBytes, 0),
    [idbRows]
  );

  const selectedItems = useMemo(() => {
    const sizeByKey = new Map();
    idbRows.forEach((r) => sizeByKey.set(rowId('indexeddb', r.key), r.sizeBytes));
    lsRows.forEach((r) => sizeByKey.set(rowId('local', r.key), r.sizeBytes));
    return [...selected].map((id) => {
      const { storage, key } = parseRowId(id);
      return { storage, key, sizeBytes: sizeByKey.get(id) || 0 };
    });
  }, [selected, idbRows, lsRows]);

  const selectedBytes = useMemo(
    () => selectedItems.reduce((s, i) => s + i.sizeBytes, 0),
    [selectedItems]
  );

  const toggleRow = (storage, key) => {
    const id = rowId(storage, key);
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAllFilteredIdb = () => {
    setSelected((prev) => {
      const next = new Set(prev);
      filteredIdb.forEach((row) => next.add(rowId('indexeddb', row.key)));
      return next;
    });
  };

  const selectAllFilteredLs = () => {
    setSelected((prev) => {
      const next = new Set(prev);
      filteredLs.forEach((row) => next.add(rowId('local', row.key)));
      return next;
    });
  };

  const clearSelection = () => setSelected(new Set());

  const allFilteredIdbSelected =
    filteredIdb.length > 0
    && filteredIdb.every((row) => selected.has(rowId('indexeddb', row.key)));

  const allFilteredLsSelected =
    filteredLs.length > 0
    && filteredLs.every((row) => selected.has(rowId('local', row.key)));

  const deleteKeys = async (items) => {
    if (!items?.length || deleting) return;
    setDeleting(true);
    setStatusNote('');
    try {
      for (const { storage, key } of items) {
        if (storage === 'indexeddb') {
          await indexedDbStorage.removeItem(key);
        } else {
          try {
            localStorage.removeItem(key);
          } catch (e) {
            console.warn(e);
          }
        }
        notifyStorageChanged(key);
      }
      setSelected((prev) => {
        const next = new Set(prev);
        items.forEach(({ storage, key }) => next.delete(rowId(storage, key)));
        return next;
      });
      const bytes = items.reduce((s, i) => s + (i.sizeBytes || 0), 0);
      setStatusNote(`Removed ${items.length} key(s)${bytes ? ` (${formatBytes(bytes)})` : ''}.`);
      await load();
    } finally {
      setDeleting(false);
    }
  };

  const deleteKey = (key, storage) => deleteKeys([{ storage, key }]);

  const deleteSelected = () => deleteKeys(selectedItems);

  const deleteVariantsBulk = async () => {
    const keys = await indexedDbStorage.getKeysByPrefix('LiangLocal-variants-');
    if (keys.length === 0) {
      setStatusNote('No message-variant caches found.');
      return;
    }
    await deleteKeys(keys.map((key) => ({ storage: 'indexeddb', key })));
  };

  const emergencyRecoverChats = async () => {
    if (deleting) return;
    setDeleting(true);
    setStatusNote('');
    try {
      const result = await emergencyRecoverAllConversations();
      const { recovered, clearedBans, catalogWritten, shardKeyCount } = result;
      if (recovered > 0 && catalogWritten) {
        setStatusNote(
          `Recovered ${recovered} chat(s) from ${shardKeyCount} shard(s), cleared ${clearedBans} ban key(s). Reloading…`
        );
        window.location.reload();
      } else if (recovered > 0 && !catalogWritten) {
        setStatusNote('Recovered tabs but catalog write failed — see console.');
      } else {
        setStatusNote(
          shardKeyCount > 0
            ? `Found ${shardKeyCount} shard(s) but none recoverable (bans or empty data). See console.`
            : 'No recoverable chats found in this browser.'
        );
      }
    } catch (e) {
      console.error(e);
      setStatusNote('Emergency recover failed. See console.');
    } finally {
      setDeleting(false);
    }
  };

  const recoverChatsFromShardsHandler = async () => {
    if (deleting) return;
    setDeleting(true);
    setStatusNote('');
    try {
      const n = await recoverChatsFromShards();
      if (n > 0) {
        setStatusNote(`Recovered ${n} chat(s). Reloading…`);
        window.location.reload();
      } else {
        setStatusNote('No recoverable message data (or bans still active — try Emergency recover).');
      }
    } catch (e) {
      console.error(e);
      setStatusNote('Recovery failed. See console.');
    } finally {
      setDeleting(false);
    }
  };

  const repairGhostChats = async () => {
    if (deleting) return;
    setDeleting(true);
    setStatusNote('');
    try {
      const result = await repairAndPurgeGhostChats();
      if (result?.skipped) {
        setStatusNote('Repair skipped — catalog empty. Run Emergency recover first.');
        return;
      }
      setStatusNote('Orphan shards purged. Reloading…');
      window.location.reload();
    } catch (e) {
      console.error(e);
      setStatusNote('Repair failed — see console.');
    } finally {
      setDeleting(false);
    }
  };

  const eraseAllChats = async () => {
    if (deleting) return;
    setDeleting(true);
    setStatusNote('');
    try {
      const { deleteAllConversationsFromStorage } = await import('../utils/conversationStorage');
      await deleteAllConversationsFromStorage();
      window.location.reload();
    } catch (e) {
      console.error(e);
      setStatusNote('Failed to clear some data. See console.');
    } finally {
      setDeleting(false);
    }
  };

  const kindOptions = useMemo(() => {
    const set = new Set(idbRows.map((r) => r.kind));
    return ['all', ...Array.from(set).sort()];
  }, [idbRows]);

  return (
    <div className="space-y-4">
      {quota?.quota != null && (
        <div className="rounded-lg border border-border/60 bg-background/40 px-4 py-3 text-sm">
          <div className="flex justify-between gap-2 text-muted-foreground mb-1">
            <span>Browser storage (this origin)</span>
            <span className="tabular-nums">
              {formatBytes(quota.usage)} / {formatBytes(quota.quota)}
            </span>
          </div>
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full bg-primary/80 transition-all"
              style={{
                width: `${Math.min(100, ((quota.usage || 0) / (quota.quota || 1)) * 100)}%`
              }}
            />
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            IndexedDB rows below are only the LiangLocal database; total usage includes caches and other origins data.
          </p>
        </div>
      )}

      <div className="flex flex-col sm:flex-row gap-3 sm:items-end">
        <div className="flex-1 space-y-1">
          <Label htmlFor="storage-filter">Filter keys</Label>
          <Input
            id="storage-filter"
            placeholder="Substring match…"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />
        </div>
        <div className="w-full sm:w-52 space-y-1">
          <Label htmlFor="storage-kind">Category</Label>
          <Select value={kindFilter} onValueChange={setKindFilter}>
            <SelectTrigger id="storage-kind">
              <SelectValue placeholder="Kind" />
            </SelectTrigger>
            <SelectContent>
              {kindOptions.map((k) => (
                <SelectItem key={k} value={k}>
                  {k === 'all' ? 'All categories' : KIND_LABEL[k] || k}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button type="button" variant="outline" onClick={load} disabled={loading || deleting}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          <span className="ml-2">Refresh</span>
        </Button>
      </div>

      {selected.size > 0 ? (
        <div className="flex flex-wrap gap-2 items-center rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2">
          <span className="text-sm">
            {selected.size} selected ({formatBytes(selectedBytes)})
          </span>
          <Button
            type="button"
            variant="destructive"
            size="sm"
            disabled={deleting}
            onClick={deleteSelected}
          >
            {deleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            <span className="ml-2">Delete selected</span>
          </Button>
          <Button type="button" variant="ghost" size="sm" disabled={deleting} onClick={clearSelection}>
            Clear selection
          </Button>
        </div>
      ) : null}

      {statusNote ? (
        <p className="text-sm text-muted-foreground rounded-lg border border-border/60 bg-background/40 px-3 py-2">
          {statusNote}
        </p>
      ) : null}

      <div className="flex flex-wrap gap-2 items-center">
        <Button type="button" variant="secondary" size="sm" onClick={deleteVariantsBulk} disabled={deleting}>
          Delete all message-variant caches
        </Button>
        <Button type="button" variant="default" size="sm" onClick={emergencyRecoverChats} disabled={deleting}>
          Emergency recover chats
        </Button>
        <Button type="button" variant="outline" size="sm" onClick={recoverChatsFromShardsHandler} disabled={deleting}>
          Recover chats from shards
        </Button>
        <Button type="button" variant="outline" size="sm" onClick={repairGhostChats} disabled={deleting}>
          Repair ghost chats
        </Button>
        <Button type="button" variant="outline" size="sm" onClick={eraseAllChats} disabled={deleting} className="border-destructive/50 text-destructive hover:bg-destructive/10">
          Erase all conversations
        </Button>
        <span className="text-xs text-muted-foreground">
          Variants are safe to drop. Total IndexedDB (sum of keys): ~{formatBytes(totalIndexedBytes)}.
        </span>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading storage inventory…
        </p>
      ) : (
        <>
          <div className="rounded-lg border border-border/60 overflow-hidden">
            <div className="max-h-[min(420px,50vh)] overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-muted/80 backdrop-blur-sm border-b">
                  <tr className="text-left">
                    <th className="px-3 py-2 font-medium w-10">
                      <Checkbox
                        checked={allFilteredIdbSelected}
                        onCheckedChange={(checked) => {
                          if (checked) selectAllFilteredIdb();
                          else {
                            setSelected((prev) => {
                              const next = new Set(prev);
                              filteredIdb.forEach((row) => next.delete(rowId('indexeddb', row.key)));
                              return next;
                            });
                          }
                        }}
                        aria-label="Select all visible IndexedDB keys"
                      />
                    </th>
                    <th className="px-3 py-2 font-medium">Key</th>
                    <th className="px-3 py-2 font-medium w-28">Size</th>
                    <th className="px-3 py-2 font-medium">Category</th>
                    <th className="px-3 py-2 font-medium w-24 text-right"> </th>
                  </tr>
                </thead>
                <tbody>
                  {filteredIdb.map((row) => {
                    const id = rowId('indexeddb', row.key);
                    const isSelected = selected.has(id);
                    return (
                      <tr
                        key={row.key}
                        className={`border-b border-border/40 hover:bg-muted/30 ${isSelected ? 'bg-muted/40' : ''}`}
                      >
                        <td className="px-3 py-2 align-top">
                          <Checkbox
                            checked={isSelected}
                            onCheckedChange={() => toggleRow('indexeddb', row.key)}
                            aria-label={`Select ${row.key}`}
                          />
                        </td>
                        <td className="px-3 py-2 align-top break-all font-mono text-xs">
                          {row.key}
                          {row.convId ? (
                            <span className="block text-[11px] text-muted-foreground mt-0.5">
                              {convNameById.get(row.convId) ? `Chat: ${convNameById.get(row.convId)}` : `id: ${row.convId}`}
                            </span>
                          ) : null}
                        </td>
                        <td className="px-3 py-2 align-top tabular-nums whitespace-nowrap">{formatBytes(row.sizeBytes)}</td>
                        <td className="px-3 py-2 align-top text-xs text-muted-foreground">
                          {KIND_LABEL[row.kind] || row.kind}
                        </td>
                        <td className="px-3 py-2 align-top text-right">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="text-destructive hover:text-destructive"
                            disabled={deleting}
                            onClick={() => deleteKey(row.key, 'indexeddb')}
                            title="Delete"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {filteredIdb.length === 0 ? (
                <p className="px-3 py-6 text-center text-sm text-muted-foreground">No IndexedDB keys match.</p>
              ) : null}
            </div>
          </div>

          {lsRows.length > 0 ? (
            <div className="space-y-2">
              <p className="text-sm font-medium">localStorage only (not mirrored to IndexedDB)</p>
              <div className="rounded-lg border border-border/60 overflow-hidden">
                <div className="max-h-[min(220px,35vh)] overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-muted/80 backdrop-blur-sm border-b">
                      <tr className="text-left">
                        <th className="px-3 py-2 font-medium w-10">
                          <Checkbox
                            checked={allFilteredLsSelected}
                            onCheckedChange={(checked) => {
                              if (checked) selectAllFilteredLs();
                              else {
                                setSelected((prev) => {
                                  const next = new Set(prev);
                                  filteredLs.forEach((row) => next.delete(rowId('local', row.key)));
                                  return next;
                                });
                              }
                            }}
                            aria-label="Select all visible localStorage keys"
                          />
                        </th>
                        <th className="px-3 py-2 font-medium">Key</th>
                        <th className="px-3 py-2 font-medium w-28">Size</th>
                        <th className="px-3 py-2 font-medium w-24 text-right"> </th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredLs.map((row) => {
                        const id = rowId('local', row.key);
                        const isSelected = selected.has(id);
                        return (
                          <tr
                            key={row.key}
                            className={`border-b border-border/40 hover:bg-muted/30 ${isSelected ? 'bg-muted/40' : ''}`}
                          >
                            <td className="px-3 py-2 align-top">
                              <Checkbox
                                checked={isSelected}
                                onCheckedChange={() => toggleRow('local', row.key)}
                                aria-label={`Select ${row.key}`}
                              />
                            </td>
                            <td className="px-3 py-2 align-top break-all font-mono text-xs">{row.key}</td>
                            <td className="px-3 py-2 align-top tabular-nums">{formatBytes(row.sizeBytes)}</td>
                            <td className="px-3 py-2 align-top text-right">
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="text-destructive hover:text-destructive"
                                disabled={deleting}
                                onClick={() => deleteKey(row.key, 'local')}
                                title="Delete"
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
