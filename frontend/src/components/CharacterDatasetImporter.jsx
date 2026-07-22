import React, { useCallback, useMemo, useRef, useState } from 'react';
import { ArrowLeft, FileUp, Loader2, Search, Users } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Badge } from './ui/badge';
import {
  CHARACTER_DATASET_FIELDS,
  autoMapCharacterColumns,
  characterFromDatasetRow,
  parseCharacterDatasetText,
} from '../utils/characterDatasetImport';

const CharacterDatasetImporter = ({ onImport, onCancel }) => {
  const { PRIMARY_API_URL } = useApp();
  const fileRef = useRef(null);
  const [source, setSource] = useState('huggingface');
  const [repoId, setRepoId] = useState('');
  const [rows, setRows] = useState([]);
  const [columns, setColumns] = useState([]);
  const [mapping, setMapping] = useState({});
  const [selected, setSelected] = useState(new Set());
  const [sourceMeta, setSourceMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadRows = useCallback((nextRows, meta) => {
    const cleanRows = nextRows.filter((row) => row && typeof row === 'object' && !Array.isArray(row)).slice(0, 100);
    const nextColumns = [...new Set(cleanRows.flatMap((row) => Object.keys(row)))].sort();
    setRows(cleanRows);
    setColumns(nextColumns);
    setMapping(autoMapCharacterColumns(nextColumns));
    setSelected(new Set(cleanRows.map((_, index) => index)));
    setSourceMeta(meta);
  }, []);

  const inspectHuggingFace = useCallback(async () => {
    if (!repoId.trim()) return;
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${PRIMARY_API_URL}/character-datasets/huggingface/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_id: repoId.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not preview that dataset.');
      loadRows(data.rows || [], { type: 'huggingface', repo_id: data.repo_id, config: data.config, split: data.split });
    } catch (previewError) {
      setError(previewError.message);
    } finally {
      setLoading(false);
    }
  }, [PRIMARY_API_URL, loadRows, repoId]);

  const handleFile = useCallback(async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const parsed = parseCharacterDatasetText(await file.text(), file.name);
      if (!parsed.length) throw new Error('No character rows were found in that file.');
      loadRows(parsed, { type: 'file', name: file.name });
    } catch (fileError) {
      setError(`Mirid could not read this dataset: ${fileError.message}`);
    } finally {
      setLoading(false);
      event.target.value = '';
    }
  }, [loadRows]);

  const previews = useMemo(
    () => rows.map((row) => characterFromDatasetRow(row, mapping, sourceMeta || {})),
    [mapping, rows, sourceMeta],
  );
  const validSelected = previews.filter((preview, index) => selected.has(index) && preview.valid);

  const toggleSelected = (index) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(index)) next.delete(index); else next.add(index);
      return next;
    });
  };

  return (
    <div className="space-y-5 pb-20">
      <div className="flex flex-col gap-3 rounded-2xl border border-border/70 bg-card/60 p-5 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">Dataset Import</p>
          <h2 className="mt-1 text-2xl font-semibold">Bring a character collection into focus.</h2>
          <p className="mt-2 max-w-3xl text-sm text-muted-foreground">Preview the rows, tell Mirid what each column means, then choose exactly which characters enter your library.</p>
        </div>
        <Button variant="ghost" onClick={onCancel}><ArrowLeft className="mr-2 h-4 w-4" />Character Library</Button>
      </div>

      {error && <Alert variant="destructive"><AlertTitle>Dataset import stopped</AlertTitle><AlertDescription>{error}</AlertDescription></Alert>}

      <div className="rounded-2xl border border-border/70 bg-card/60 p-5 space-y-4">
        <div className="flex w-fit rounded-lg border bg-background/50 p-1">
          <Button size="sm" variant={source === 'huggingface' ? 'default' : 'ghost'} onClick={() => setSource('huggingface')}>Hugging Face</Button>
          <Button size="sm" variant={source === 'file' ? 'default' : 'ghost'} onClick={() => setSource('file')}>Local file</Button>
        </div>
        {source === 'huggingface' ? (
          <div className="flex flex-col gap-2 md:flex-row">
            <Input value={repoId} onChange={(event) => setRepoId(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') inspectHuggingFace(); }} placeholder="owner/dataset" className="font-mono" />
            <Button onClick={inspectHuggingFace} disabled={!repoId.trim() || loading}>{loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}Preview dataset</Button>
          </div>
        ) : (
          <div>
            <input ref={fileRef} type="file" accept=".json,.jsonl,.ndjson,.csv" className="hidden" onChange={handleFile} />
            <Button variant="outline" onClick={() => fileRef.current?.click()} disabled={loading}><FileUp className="mr-2 h-4 w-4" />Choose JSON, JSONL or CSV</Button>
            <p className="mt-2 text-xs text-muted-foreground">For Parquet, use its Hugging Face repository. Mirid can preview the hosted split without installing a data-science runtime.</p>
          </div>
        )}
      </div>

      {rows.length > 0 && (
        <>
          <div className="rounded-2xl border border-border/70 bg-card/60 p-5 space-y-4">
            <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
              <div><h3 className="font-semibold">Match the columns</h3><p className="mt-1 text-xs text-muted-foreground">Mirid has made a first pass. Only a name and some usable character content are required.</p></div>
              {sourceMeta?.repo_id && <Badge variant="outline">{sourceMeta.repo_id} · {sourceMeta.config}/{sourceMeta.split}</Badge>}
            </div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {CHARACTER_DATASET_FIELDS.map(([field, label]) => (
                <div key={field} className="space-y-1.5">
                  <label className="text-xs font-medium">{label}</label>
                  <Select value={mapping[field] || '__none__'} onValueChange={(value) => setMapping((current) => ({ ...current, [field]: value === '__none__' ? '' : value }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent><SelectItem value="__none__">Not mapped</SelectItem>{columns.map((column) => <SelectItem key={column} value={column}>{column}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-border/70 bg-card/60 p-5 space-y-4">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div><h3 className="font-semibold">Choose characters</h3><p className="mt-1 text-xs text-muted-foreground">Showing the first {rows.length} rows. Invalid rows remain visible but cannot be imported.</p></div>
              <Button onClick={() => onImport(validSelected.map((preview) => preview.character))} disabled={validSelected.length === 0}><Users className="mr-2 h-4 w-4" />Import {validSelected.length} character{validSelected.length === 1 ? '' : 's'}</Button>
            </div>
            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
              {previews.map((preview, index) => (
                <label key={index} className={`flex cursor-pointer gap-3 rounded-xl border p-3 ${preview.valid ? 'border-border/60 bg-background/40' : 'border-destructive/40 bg-destructive/5'}`}>
                  <input type="checkbox" className="mt-1" checked={selected.has(index) && preview.valid} disabled={!preview.valid} onChange={() => toggleSelected(index)} />
                  <div className="min-w-0"><p className="truncate text-sm font-medium">{preview.character.name || `Row ${index + 1}`}</p><p className="mt-1 line-clamp-3 text-xs leading-relaxed text-muted-foreground">{preview.character.description || preview.character.first_message || 'No usable character description was mapped.'}</p>{!preview.valid && <p className="mt-2 text-[11px] text-destructive">Needs a name and character content.</p>}</div>
                </label>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default CharacterDatasetImporter;
