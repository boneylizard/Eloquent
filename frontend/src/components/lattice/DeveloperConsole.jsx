import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Terminal, RefreshCw, Trash2, Play, ChevronDown, ChevronRight, Eye, EyeOff, Edit3, Save, X, Send, Loader2, AlertTriangle, CheckCircle, Clock, Filter } from 'lucide-react';
import { usePool } from '../../contexts/PoolContext';
import { getBackendUrl } from '../../config/api';

const API_BASE = getBackendUrl();

function JsonViewer({ data, onEdit, editable = false }) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState('');
  const [error, setError] = useState(null);

  const startEdit = () => {
    setEditText(JSON.stringify(data, null, 2));
    setEditing(true);
    setError(null);
  };

  const saveEdit = () => {
    try {
      const parsed = JSON.parse(editText);
      onEdit(parsed);
      setEditing(false);
      setError(null);
    } catch (e) {
      setError('Invalid JSON: ' + e.message);
    }
  };

  if (editing) {
    return (
      <div className="space-y-2">
        {error && <div className="text-xs text-red-400 bg-red-500/10 px-2 py-1 rounded">{error}</div>}
        <textarea
          value={editText}
          onChange={e => setEditText(e.target.value)}
          className="w-full h-64 text-[11px] font-mono bg-muted border rounded-lg p-2 outline-none resize-y"
          spellCheck={false}
        />
        <div className="flex gap-2">
          <button onClick={saveEdit} className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-primary/10 text-primary hover:bg-primary/20">
            <Save className="w-3 h-3" /> Save
          </button>
          <button onClick={() => setEditing(false)} className="flex items-center gap-1 text-xs px-2 py-1 rounded text-muted-foreground hover:text-foreground">
            <X className="w-3 h-3" /> Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative group">
      <pre className="text-[11px] font-mono bg-muted/50 border rounded-lg p-3 overflow-auto max-h-80 whitespace-pre-wrap">
        {JSON.stringify(data, null, 2)}
      </pre>
      {editable && onEdit && (
        <button
          onClick={startEdit}
          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-background border text-muted-foreground hover:text-foreground"
        >
          <Edit3 className="w-2.5 h-2.5" /> Edit
        </button>
      )}
    </div>
  );
}

function InteractionLogPanel({ poolCharacters }) {
  const [logs, setLogs] = useState({});
  const [loading, setLoading] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  const [editingId, setEditingId] = useState(null);

  const fetchLogs = useCallback(async () => {
    setLoading(true);
    const result = {};
    for (const char of poolCharacters) {
      try {
        const resp = await fetch(`${API_BASE}/lattice/interaction-log/${encodeURIComponent(char.id)}/context?limit=999`);
        const data = await resp.json();
        if (data.status === 'success') {
          result[char.id] = data.context;
        }
      } catch (e) {
        result[char.id] = { error: e.message };
      }
    }
    setLogs(result);
    setLoading(false);
  }, [poolCharacters]);

  useEffect(() => { fetchLogs(); }, [fetchLogs]);

  const handleSaveLog = useCallback(async (charId, newData) => {
    try {
      const resp = await fetch(`${API_BASE}/lattice/interaction-log/${encodeURIComponent(charId)}`, {
        method: 'DELETE',
      });
      if (resp.ok) {
        for (const entry of (newData.interactions || [])) {
          await fetch(`${API_BASE}/lattice/interaction-log`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              character_id: charId,
              character_name: newData.character_name || '',
              entry_type: entry.type || 'exchange',
              surface: entry.surface || 'chat',
              actor: entry.actor || 'user',
              user_message: entry.user_message || '',
              character_response: entry.character_response || '',
              content: entry.content || '',
              emotional_state: entry.emotional_state,
              target_character: entry.target_character,
              context: entry.context,
            }),
          });
        }
        fetchLogs();
      }
    } catch (e) {
      console.warn('Failed to save log:', e);
    }
  }, [fetchLogs]);

  const handleDeleteLog = useCallback(async (charId) => {
    if (!window.confirm('Delete this interaction log?')) return;
    try {
      await fetch(`${API_BASE}/lattice/interaction-log/${encodeURIComponent(charId)}`, { method: 'DELETE' });
      fetchLogs();
    } catch (e) {
      console.warn('Failed to delete log:', e);
    }
  }, [fetchLogs]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Interaction Logs</h3>
        <button onClick={fetchLogs} disabled={loading} className="flex items-center gap-1 text-xs px-2 py-1 rounded border text-muted-foreground hover:text-foreground disabled:opacity-50">
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>
      {poolCharacters.length === 0 && <p className="text-xs text-muted-foreground">No characters in pool.</p>}
      {poolCharacters.map(char => {
        const log = logs[char.id];
        const isExpanded = expandedId === char.id;
        const entryCount = log?.interactions?.length || log?.raw_count || 0;
        const totalChars = log?.total_chars || 0;
        return (
          <div key={char.id} className="border rounded-lg overflow-hidden">
            <button
              onClick={() => setExpandedId(isExpanded ? null : char.id)}
              className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-muted/30 transition-colors"
            >
              {isExpanded ? <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />}
              {char.avatar ? (
                <img src={char.avatar} alt="" className="w-6 h-6 rounded-full object-cover" />
              ) : (
                <div className="w-6 h-6 rounded-full bg-muted flex items-center justify-center text-[9px] font-bold">{char.name?.[0]}</div>
              )}
              <span className="text-xs font-medium flex-1">{char.name}</span>
              <span className="text-[10px] text-muted-foreground">{entryCount} entries · {totalChars} chars</span>
              <button
                onClick={(e) => { e.stopPropagation(); handleDeleteLog(char.id); }}
                className="opacity-0 group-hover:opacity-100 text-muted-foreground/40 hover:text-red-400 transition-colors p-1"
                title="Delete log"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </button>
            {isExpanded && log && !log.error && (
              <div className="px-3 pb-3 space-y-2 border-t border-border/30 pt-2">
                <div className="flex gap-4 text-[10px] text-muted-foreground">
                  <span>Entries: {entryCount}</span>
                  <span>Total chars: {totalChars}</span>
                  <span>Compacted: {log.compacted_at ? 'Yes' : 'No'}</span>
                </div>
                {log.compacted_summary && (
                  <div>
                    <div className="text-[10px] font-semibold text-muted-foreground mb-1">Compacted Summary</div>
                    <div className="text-[11px] bg-muted/50 rounded p-2 whitespace-pre-wrap">{log.compacted_summary}</div>
                  </div>
                )}
                <div>
                  <div className="text-[10px] font-semibold text-muted-foreground mb-1">Raw Log Data</div>
                  <JsonViewer data={log} editable onEdit={(d) => handleSaveLog(char.id, d)} />
                </div>
                {log.formatted_text && (
                  <div>
                    <div className="text-[10px] font-semibold text-muted-foreground mb-1">Formatted (what the AI sees)</div>
                    <pre className="text-[11px] font-mono bg-muted/50 border rounded-lg p-3 whitespace-pre-wrap max-h-60 overflow-auto">
                      {log.formatted_text}
                    </pre>
                  </div>
                )}
              </div>
            )}
            {isExpanded && log?.error && (
              <div className="px-3 pb-3 text-xs text-red-400">{log.error}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ApiTester() {
  const [method, setMethod] = useState('GET');
  const [endpoint, setEndpoint] = useState('/lattice/feed');
  const [body, setBody] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const sendRequest = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResponse(null);
    const start = Date.now();
    try {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (body && method !== 'GET') {
        try { JSON.parse(body); opts.body = body; } catch { setError('Invalid JSON body'); setLoading(false); return; }
      }
      const resp = await fetch(`${API_BASE}${endpoint}`, opts);
      const elapsed = Date.now() - start;
      const data = await resp.json();
      setResponse({ status: resp.status, elapsed, data });
      setHistory(prev => [{ method, endpoint, status: resp.status, elapsed, time: new Date().toLocaleTimeString() }, ...prev.slice(0, 19)]);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }, [method, endpoint, body]);

  const presets = [
    { label: 'GET Feed', method: 'GET', endpoint: '/lattice/feed' },
    { label: 'GET DM Threads', method: 'GET', endpoint: '/lattice/dm-threads' },
    { label: 'GET Stories', method: 'GET', endpoint: '/lattice/stories' },
    { label: 'GET Pool State', method: 'GET', endpoint: '/lattice/pool-state' },
    { label: 'GET Voice List', method: 'GET', endpoint: '/lattice/voice-list' },
    { label: 'POST Agentic Tick', method: 'POST', endpoint: '/lattice/agentic-tick', body: '{\n  "model_name": "",\n  "actor_type": "female_ai",\n  "action_type": "full",\n  "character_name": "Test",\n  "character_profile": {},\n  "memory_entries": [],\n  "pool_summary": "",\n  "dummy_activity": "",\n  "dummy_realism": 50,\n    "dummy_agency": 50\n}' },
  ];

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold">API Tester</h3>
      <div className="flex gap-2">
        <select value={method} onChange={e => setMethod(e.target.value)} className="h-8 text-xs bg-muted border rounded px-2">
          <option>GET</option><option>POST</option><option>PUT</option><option>DELETE</option>
        </select>
        <input value={endpoint} onChange={e => setEndpoint(e.target.value)} placeholder="/lattice/..." className="flex-1 h-8 text-xs bg-muted border rounded px-2 font-mono" />
        <button onClick={sendRequest} disabled={loading} className="flex items-center gap-1 h-8 text-xs px-3 rounded bg-primary/10 text-primary hover:bg-primary/20 disabled:opacity-50">
          {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Send className="w-3 h-3" />} Send
        </button>
      </div>
      {method !== 'GET' && (
        <textarea value={body} onChange={e => setBody(e.target.value)} placeholder="JSON body..." className="w-full h-24 text-[11px] font-mono bg-muted border rounded p-2 resize-y" spellCheck={false} />
      )}
      <div className="flex gap-1 flex-wrap">
        {presets.map(p => (
          <button key={p.label} onClick={() => { setMethod(p.method); setEndpoint(p.endpoint); setBody(p.body || ''); }} className="text-[10px] px-2 py-1 rounded border text-muted-foreground hover:text-foreground hover:bg-muted">
            {p.label}
          </button>
        ))}
      </div>
      {error && <div className="text-xs text-red-400 bg-red-500/10 px-2 py-1 rounded">{error}</div>}
      {response && (
        <div className="space-y-2">
          <div className="flex gap-3 text-[10px] text-muted-foreground">
            <span className={response.status < 400 ? 'text-green-400' : 'text-red-400'}>Status: {response.status}</span>
            <span>Time: {response.elapsed}ms</span>
          </div>
          <JsonViewer data={response.data} />
        </div>
      )}
      {history.length > 0 && (
        <div>
          <div className="text-[10px] font-semibold text-muted-foreground mb-1">History</div>
          <div className="space-y-1">
            {history.map((h, i) => (
              <div key={i} className="flex items-center gap-2 text-[10px] text-muted-foreground">
                <span className="font-mono">{h.time}</span>
                <span className={`font-semibold ${h.status < 400 ? 'text-green-400' : 'text-red-400'}`}>{h.method}</span>
                <span className="font-mono truncate flex-1">{h.endpoint}</span>
                <span>{h.status}</span>
                <span>{h.elapsed}ms</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function EventLogPanel() {
  const { activityLog, agenticActionLog } = usePool();
  const [filter, setFilter] = useState('all');
  const [expanded, setExpanded] = useState(false);

  const allEvents = [
    ...(activityLog || []).map(e => ({ ...e, source: 'activity' })),
    ...(agenticActionLog || []).map(e => ({ ...e, source: 'agentic', type: 'agentic_action', detail: `${e.action}: ${e.content || e.reasoning || ''}` })),
  ].sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));

  const filtered = filter === 'all' ? allEvents : allEvents.filter(e => e.source === filter || e.type === filter);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Event Log</h3>
        <div className="flex gap-1">
          {['all', 'activity', 'agentic'].map(f => (
            <button key={f} onClick={() => setFilter(f)} className={`text-[10px] px-2 py-1 rounded ${filter === f ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'}`}>
              {f}
            </button>
          ))}
        </div>
      </div>
      <div className="text-[10px] text-muted-foreground">{filtered.length} events</div>
      <div className="space-y-1 max-h-96 overflow-y-auto">
        {filtered.slice(0, 100).map((e, i) => (
          <div key={i} className="flex items-start gap-2 text-[10px] py-1 border-b border-border/20">
            <span className="text-muted-foreground font-mono shrink-0 w-16">{e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : ''}</span>
            <span className={`shrink-0 font-semibold ${e.source === 'agentic' ? 'text-amber-400' : 'text-emerald-400'}`}>{e.source}</span>
            <span className="text-muted-foreground shrink-0">{e.type || e.action}</span>
            {e.character && <span className="text-primary shrink-0">{e.character}</span>}
            <span className="flex-1 min-w-0 truncate">{e.detail || e.content || ''}</span>
            {e.success === false && <AlertTriangle className="w-3 h-3 text-red-400 shrink-0" />}
          </div>
        ))}
      </div>
    </div>
  );
}

function SystemStatePanel() {
  const { poolCharacters, feedPosts, dmThreads, stories, mirrorEnabled, tickEnabled, dummyRealism, dummyAgency } = usePool();

  const stats = [
    { label: 'Characters', value: poolCharacters.length },
    { label: 'Feed Posts', value: feedPosts.length },
    { label: 'DM Threads', value: dmThreads.length },
    { label: 'Stories', value: stories.length },
    { label: 'Mirror Enabled', value: mirrorEnabled ? 'Yes' : 'No' },
    { label: 'Auto-Tick', value: tickEnabled ? 'Yes' : 'No' },
    { label: 'Dummy Realism', value: dummyRealism },
    { label: 'Dummy Agency', value: dummyAgency },
  ];

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold">System State</h3>
      <div className="grid grid-cols-2 gap-2">
        {stats.map(s => (
          <div key={s.label} className="bg-muted/30 border rounded-lg px-3 py-2">
            <div className="text-[10px] text-muted-foreground">{s.label}</div>
            <div className="text-sm font-semibold">{s.value}</div>
          </div>
        ))}
      </div>
      <div>
        <div className="text-[10px] font-semibold text-muted-foreground mb-1">Characters in Pool</div>
        <div className="space-y-1">
          {poolCharacters.map(c => (
            <div key={c.id} className="flex items-center gap-2 text-xs">
              {c.avatar ? (
                <img src={c.avatar} alt="" className="w-5 h-5 rounded-full object-cover" />
              ) : (
                <div className="w-5 h-5 rounded-full bg-muted flex items-center justify-center text-[8px] font-bold">{c.name?.[0]}</div>
              )}
              <span className="font-medium">{c.name}</span>
              <span className="text-muted-foreground text-[10px]">{c.id?.slice(0, 12)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function DeveloperConsole() {
  const [activePanel, setActivePanel] = useState('logs');
  const { poolCharacters } = usePool();

  const panels = [
    { id: 'logs', label: 'Interaction Logs' },
    { id: 'api', label: 'API Tester' },
    { id: 'events', label: 'Event Log' },
    { id: 'state', label: 'System State' },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Terminal className="w-4 h-4 text-emerald-400" />
        <h2 className="text-sm font-bold">Developer Console</h2>
      </div>
      <div className="flex gap-1 bg-muted/40 rounded-lg p-0.5">
        {panels.map(p => (
          <button
            key={p.id}
            onClick={() => setActivePanel(p.id)}
            className={`text-xs px-3 py-1.5 rounded-md transition-all ${activePanel === p.id ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className="min-h-0">
        {activePanel === 'logs' && <InteractionLogPanel poolCharacters={poolCharacters} />}
        {activePanel === 'api' && <ApiTester />}
        {activePanel === 'events' && <EventLogPanel />}
        {activePanel === 'state' && <SystemStatePanel />}
      </div>
    </div>
  );
}
