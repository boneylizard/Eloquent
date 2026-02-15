import React, { useState, useMemo } from 'react';
import { getBackendUrl } from '../config/api';
import { useApp } from '../contexts/AppContext';
import './ElectionTracker.css';

const HEX_GRID = [
    { id: "AK", r: 0, c: 0 }, { id: "ME", r: 0, c: 11 },
    { id: "VT", r: 1, c: 10 }, { id: "NH", r: 1, c: 11 },
    { id: "WA", r: 2, c: 1 }, { id: "ID", r: 2, c: 2 }, { id: "MT", r: 2, c: 3 }, { id: "ND", r: 2, c: 4 }, { id: "MN", r: 2, c: 5 }, { id: "IL", r: 2, c: 6 }, { id: "WI", r: 2, c: 7 }, { id: "MI", r: 2, c: 8 }, { id: "NY", r: 2, c: 9 }, { id: "MA", r: 2, c: 10 }, { id: "RI", r: 2, c: 11 },
    { id: "OR", r: 3, c: 1 }, { id: "NV", r: 3, c: 2 }, { id: "WY", r: 3, c: 3 }, { id: "SD", r: 3, c: 4 }, { id: "IA", r: 3, c: 5 }, { id: "IN", r: 3, c: 6 }, { id: "OH", r: 3, c: 7 }, { id: "PA", r: 3, c: 8 }, { id: "NJ", r: 3, c: 9 }, { id: "CT", r: 3, c: 10 },
    { id: "CA", r: 4, c: 1 }, { id: "UT", r: 4, c: 2 }, { id: "CO", r: 4, c: 3 }, { id: "NE", r: 4, c: 4 }, { id: "MO", r: 4, c: 5 }, { id: "KY", r: 4, c: 6 }, { id: "WV", r: 4, c: 7 }, { id: "VA", r: 4, c: 8 }, { id: "MD", r: 4, c: 9 }, { id: "DE", r: 4, c: 10 }, { id: "DC", r: 4, c: 11 },
    { id: "AZ", r: 5, c: 2 }, { id: "NM", r: 5, c: 3 }, { id: "KS", r: 5, c: 4 }, { id: "AR", r: 5, c: 5 }, { id: "TN", r: 5, c: 6 }, { id: "NC", r: 5, c: 7 }, { id: "SC", r: 5, c: 8 },
    { id: "OK", r: 6, c: 4 }, { id: "LA", r: 6, c: 5 }, { id: "MS", r: 6, c: 6 }, { id: "AL", r: 6, c: 7 }, { id: "GA", r: 6, c: 8 },
    { id: "HI", r: 7, c: 1 }, { id: "TX", r: 7, c: 4 }, { id: "FL", r: 7, c: 9 },
];

const ElectionMap = ({ raceType = 'senate', userData = [], scrapedAverages = {}, onRefresh, winProbByState = null }) => {
    const { PRIMARY_API_URL } = useApp();
    const [editingState, setEditingState] = useState(null);
    const [expandedState, setExpandedState] = useState(null);
    const [editForm, setEditForm] = useState({
        candidate_1_name: '',
        candidate_1_party: 'D',
        candidate_1_pct: '',
        candidate_2_name: '',
        candidate_2_party: 'R',
        candidate_2_pct: '',
        margin: '',
        source_note: ''
    });

    const mapDataByState = useMemo(() => {
        const byState = {};
        userData.forEach((row) => {
            const stateKey = (row.state || '').toUpperCase().trim();
            if (!stateKey) return;
            byState[stateKey] = {
                source: 'manual',
                margin: row.margin,
                candidate_1_name: row.candidate_1_name,
                candidate_1_party: row.candidate_1_party,
                candidate_1_pct: row.candidate_1_pct,
                candidate_2_name: row.candidate_2_name,
                candidate_2_party: row.candidate_2_party,
                candidate_2_pct: row.candidate_2_pct,
                source_note: row.source_note,
            };
        });
        Object.entries(scrapedAverages).forEach(([state, avg]) => {
            const stateKey = (state || '').toUpperCase().trim();
            if (!stateKey || byState[stateKey]) return;
            const dem = avg.dem_avg;
            const gop = avg.gop_avg ?? avg.rep_avg;
            const margin = dem != null && gop != null ? dem - gop : null;
            byState[stateKey] = {
                source: 'polls',
                margin,
                candidate_1_party: 'D',
                candidate_2_party: 'R',
                dem_avg: dem,
                gop_avg: gop,
                poll_count: avg.poll_count,
                polls: Array.isArray(avg.polls) ? avg.polls : [],
                pollster_confidence: avg.pollster_confidence ?? null,
                pollster_confidence_note: avg.pollster_confidence_note ?? null,
            };
        });
        return byState;
    }, [userData, scrapedAverages]);

    const getStateTooltip = (stateId) => {
        if (winProbByState && winProbByState[stateId] != null) {
            const d = Number(winProbByState[stateId]);
            const r = 100 - d;
            return { summary: `${stateId}: D ${d.toFixed(1)}% win prob | R ${r.toFixed(1)}%`, data: { dem_win_pct: d } };
        }
        const data = mapDataByState[stateId];
        if (!data) return { summary: `${stateId}: No data. Click to enter.`, data: null };
        const margin = data.margin != null ? Number(data.margin) : NaN;
        const marginStr = Number.isNaN(margin) ? '—' : (margin > 0 ? `D+${margin.toFixed(1)}` : margin < 0 ? `R+${(-margin).toFixed(1)}` : 'Tie');
        const sourceLabel = data.source === 'polls' ? 'Polling averages' : 'Manual entry';
        let summary = `${stateId}: ${marginStr} (${sourceLabel})`;
        if (data.dem_avg != null && data.gop_avg != null) {
            summary += ` — D ${Number(data.dem_avg).toFixed(1)}% R ${Number(data.gop_avg).toFixed(1)}%`;
            if (data.poll_count) summary += ` (${data.poll_count} poll${data.poll_count !== 1 ? 's' : ''})`;
        }
        if (data.source === 'polls' && data.pollster_confidence != null) {
            const pct = Math.round(Number(data.pollster_confidence) * 100);
            summary += ` — Confidence ${pct}%`;
        }
        summary += '. Click to edit.';
        return { summary, data };
    };

    const getStateColor = (stateId) => {
        if (winProbByState && winProbByState[stateId] != null) {
            const demPct = Number(winProbByState[stateId]);
            if (demPct >= 90) return 'rgb(30, 64, 175)';
            if (demPct >= 60) return 'rgb(59, 130, 246)';
            if (demPct > 40 && demPct < 60) return '#6b7280';
            if (demPct <= 10) return 'rgb(127, 29, 29)';
            if (demPct <= 40) return 'rgb(239, 68, 68)';
            return '#6b7280';
        }
        const data = mapDataByState[stateId];
        if (!data) return '#333333';
        const margin = Number(data.margin);
        if (Number.isNaN(margin) || margin === 0) return '#8B8B00';
        const intensity = Math.min(Math.abs(margin) / 20, 1);
        const leadDem = (data.candidate_1_party === 'D' && margin > 0) || (data.candidate_2_party === 'D' && margin < 0);
        if (leadDem) {
            const r = Math.round(59 - 59 * intensity);
            const g = Math.round(130 - 80 * intensity);
            const b = 246;
            return `rgb(${r}, ${g}, ${b})`;
        }
        const r = 239;
        const g = Math.round(68 - 68 * intensity);
        const b = Math.round(68 - 68 * intensity);
        return `rgb(${r}, ${g}, ${b})`;
    };

    const openEdit = (stateId) => {
        setExpandedState(null);
        const data = mapDataByState[stateId] || {};
        setEditForm({
            candidate_1_name: data.candidate_1_name || '',
            candidate_1_party: data.candidate_1_party || 'D',
            candidate_1_pct: data.candidate_1_pct ?? '',
            candidate_2_name: data.candidate_2_name || '',
            candidate_2_party: data.candidate_2_party || 'R',
            candidate_2_pct: data.candidate_2_pct ?? '',
            margin: data.margin ?? '',
            source_note: data.source_note || ''
        });
        setEditingState(stateId);
    };

    const baseUrl = PRIMARY_API_URL || getBackendUrl();

    const saveMapData = async () => {
        if (!editingState) return;
        const marginNum = editForm.margin === '' ? null : parseFloat(editForm.margin);
        const c1pct = editForm.candidate_1_pct === '' ? null : parseFloat(editForm.candidate_1_pct);
        const c2pct = editForm.candidate_2_pct === '' ? null : parseFloat(editForm.candidate_2_pct);
        await fetch(`${baseUrl}/election/map/${editingState}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                race_type: raceType,
                candidate_1_name: editForm.candidate_1_name || null,
                candidate_1_party: editForm.candidate_1_party || null,
                candidate_1_pct: c1pct,
                candidate_2_name: editForm.candidate_2_name || null,
                candidate_2_party: editForm.candidate_2_party || null,
                candidate_2_pct: c2pct,
                margin: marginNum,
                source_note: editForm.source_note || null
            })
        });
        setEditingState(null);
        if (onRefresh) onRefresh();
    };

    const deleteMapData = async () => {
        if (!editingState) return;
        await fetch(`${baseUrl}/election/map/${editingState}?race_type=${raceType}`, { method: 'DELETE' });
        setEditingState(null);
        if (onRefresh) onRefresh();
    };

    return (
        <div className="election-map-container">
            <div className="election-map-grid">
                {HEX_GRID.map((state) => {
                    const { summary, data } = getStateTooltip(state.id);
                    return (
                        <div
                            key={state.id}
                            title={summary}
                            onClick={() => setExpandedState(state.id)}
                            style={{
                                gridColumn: state.c + 1,
                                gridRow: state.r + 1,
                                aspectRatio: '1',
                                backgroundColor: getStateColor(state.id),
                                border: editingState === state.id ? '2px solid white' : '1px solid #444',
                                borderRadius: '4px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                fontWeight: 'bold',
                                color: '#fff',
                                transition: 'all 0.2s',
                                position: 'relative'
                            }}
                            className="state-tile"
                        >
                            {state.id}
                        </div>
                    );
                })}
            </div>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginTop: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
                <span><span style={{ background: '#3b82f6', width: 12, height: 12, display: 'inline-block', borderRadius: 2, marginRight: 4 }} /> Dem lead</span>
                <span><span style={{ background: '#8B8B00', width: 12, height: 12, display: 'inline-block', borderRadius: 2, marginRight: 4 }} /> Toss-up</span>
                <span><span style={{ background: '#ef4444', width: 12, height: 12, display: 'inline-block', borderRadius: 2, marginRight: 4 }} /> GOP lead</span>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Hover for summary · Click state for details</span>
            </div>

            {expandedState && mapDataByState[expandedState] && (
                <div
                    className="map-details-panel"
                    style={{
                        marginTop: '1rem',
                        marginLeft: 'auto',
                        marginRight: 'auto',
                        maxWidth: '400px',
                        background: 'var(--bg-tertiary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '8px',
                        padding: '1rem',
                        fontSize: '0.9rem',
                        color: 'var(--text-primary)'
                    }}
                >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', flexWrap: 'wrap', gap: '6px' }}>
                        <strong>Data for {expandedState}</strong>
                        <span style={{ display: 'flex', gap: '6px' }}>
                            <button type="button" className="election-tab" style={{ fontSize: '0.75rem', padding: '2px 8px' }} onClick={() => openEdit(expandedState)}>Edit</button>
                            <button type="button" className="election-tab" style={{ fontSize: '0.75rem', padding: '2px 8px' }} onClick={() => setExpandedState(null)}>Close</button>
                        </span>
                    </div>
                    {(() => {
                        const d = mapDataByState[expandedState];
                        const margin = d.margin != null ? Number(d.margin) : null;
                        const marginStr = margin == null ? '—' : (margin > 0 ? `D+${margin.toFixed(1)}` : margin < 0 ? `R+${(-margin).toFixed(1)}` : 'Tie');
                        return (
                            <>
                                <dl style={{ margin: 0, display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '0.25rem 1rem' }}>
                                    <dt style={{ margin: 0, color: 'var(--text-secondary)' }}>Source</dt>
                                    <dd style={{ margin: 0 }}>{d.source === 'polls' ? 'Polling averages (same data as Polls tab)' : 'Manual entry'}</dd>
                                    <dt style={{ margin: 0, color: 'var(--text-secondary)' }}>Margin</dt>
                                    <dd style={{ margin: 0 }}>{marginStr}</dd>
                                    {d.source === 'polls' && (
                                        <>
                                            {d.dem_avg != null && (
                                                <>
                                                    <dt style={{ margin: 0, color: 'var(--text-secondary)' }}>Dem avg</dt>
                                                    <dd style={{ margin: 0 }}>{Number(d.dem_avg).toFixed(1)}%</dd>
                                                </>
                                            )}
                                            {d.gop_avg != null && (
                                                <>
                                                    <dt style={{ margin: 0, color: 'var(--text-secondary)' }}>GOP avg</dt>
                                                    <dd style={{ margin: 0 }}>{Number(d.gop_avg).toFixed(1)}%</dd>
                                                </>
                                            )}
                                            {d.poll_count != null && d.poll_count > 0 && (
                                                <>
                                                    <dt style={{ margin: 0, color: 'var(--text-secondary)' }}>Polls</dt>
                                                    <dd style={{ margin: 0 }}>{d.poll_count} poll{d.poll_count !== 1 ? 's' : ''} averaged</dd>
                                                </>
                                            )}
                                            {d.pollster_confidence != null && (
                                                <>
                                                    <dt style={{ margin: 0, color: 'var(--text-secondary)' }}>Confidence</dt>
                                                    <dd style={{ margin: 0 }}>{Math.round(Number(d.pollster_confidence) * 100)}% — {d.pollster_confidence_note || 'Based on FiveThirtyEight pollster ratings'}</dd>
                                                </>
                                            )}
                                        </>
                                    )}
                                    {d.source === 'manual' && (
                                        <>
                                            {(d.candidate_1_name || d.candidate_1_pct != null) && (
                                                <>
                                                    <dt style={{ margin: 0, color: 'var(--text-secondary)' }}>Candidate 1</dt>
                                                    <dd style={{ margin: 0 }}>{[d.candidate_1_name, d.candidate_1_party && `(${d.candidate_1_party})`, d.candidate_1_pct != null && `${Number(d.candidate_1_pct)}%`].filter(Boolean).join(' ')}</dd>
                                                </>
                                            )}
                                            {(d.candidate_2_name || d.candidate_2_pct != null) && (
                                                <>
                                                    <dt style={{ margin: 0, color: 'var(--text-secondary)' }}>Candidate 2</dt>
                                                    <dd style={{ margin: 0 }}>{[d.candidate_2_name, d.candidate_2_party && `(${d.candidate_2_party})`, d.candidate_2_pct != null && `${Number(d.candidate_2_pct)}%`].filter(Boolean).join(' ')}</dd>
                                                </>
                                            )}
                                        </>
                                    )}
                                    {d.source_note && (
                                        <>
                                            <dt style={{ margin: 0, color: 'var(--text-secondary)' }}>Note</dt>
                                            <dd style={{ margin: 0 }}>{d.source_note}</dd>
                                        </>
                                    )}
                                </dl>
                                {d.source === 'polls' && Array.isArray(d.polls) && d.polls.length > 0 && (
                                    <div style={{ marginTop: '0.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.75rem' }}>
                                        <strong style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Polls used for average</strong>
                                        <ul style={{ margin: '0.25rem 0 0', paddingLeft: '1.25rem', fontSize: '0.85rem' }}>
                                            {d.polls.map((poll, i) => {
                                                const diff = poll.dem_pct != null && poll.gop_pct != null ? Number(poll.dem_pct) - Number(poll.gop_pct) : null;
                                                const marginVal = poll.margin != null && poll.margin !== '' ? poll.margin : diff != null ? `${diff >= 0 ? 'D+' : 'R+'}${Math.abs(diff).toFixed(1)}` : null;
                                                const resultsStr = poll.results && typeof poll.results === 'object' && Object.keys(poll.results).length
                                                    ? Object.entries(poll.results).map(([k, v]) => `${k}: ${v}%`).join(', ')
                                                    : (poll.dem_pct != null && poll.gop_pct != null) ? `D ${poll.dem_pct}% / R ${poll.gop_pct}%` : null;
                                                return (
                                                    <li key={i} style={{ marginBottom: '0.35rem' }}>
                                                        {poll.pollster && <strong>{poll.pollster}</strong>}
                                                        {poll.race && <span> — {poll.race}</span>}
                                                        {marginVal != null && <span> ({marginVal})</span>}
                                                        {poll.added && <span style={{ color: 'var(--text-secondary)' }}> · {poll.added}</span>}
                                                        {resultsStr != null && <div style={{ marginLeft: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{resultsStr}</div>}
                                                    </li>
                                                );
                                            })}
                                        </ul>
                                    </div>
                                )}
                            </>
                        );
                    })()}
                </div>
            )}

            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '0.5rem', textAlign: 'center', maxWidth: '560px', marginLeft: 'auto', marginRight: 'auto' }}>
                States are colored from state-level polling averages (Polls tab) when available; manual entries override. Hover a state to see margin and source.
            </p>
            {(() => {
                const pollStates = Object.entries(mapDataByState).filter(([, d]) => d.source === 'polls').map(([s]) => s).sort();
                if (pollStates.length > 0) {
                    return (
                        <p style={{ color: 'var(--text-primary)', fontSize: '0.85rem', marginTop: '0.25rem', fontWeight: 600 }}>
                            {pollStates.length} state{pollStates.length !== 1 ? 's' : ''} from polls: {pollStates.join(', ')}
                        </p>
                    );
                }
                return (
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '0.25rem' }}>
                        No state-level polling data yet. Data comes from the same table as the Polls tab (STATE column).
                    </p>
                );
            })()}

            {editingState && (
                <div
                    className="map-edit-modal"
                    style={{
                        position: 'fixed',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        background: 'var(--bg-secondary)',
                        padding: '1.5rem',
                        borderRadius: '12px',
                        border: '1px solid var(--border-color)',
                        zIndex: 1000,
                        minWidth: '350px',
                        boxShadow: '0 20px 40px rgba(0,0,0,0.5)'
                    }}
                >
                    <h3>{editingState} — Enter Polling Data</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '1rem' }}>
                        <label>Candidate 1</label>
                        <input
                            placeholder="Name"
                            value={editForm.candidate_1_name}
                            onChange={(e) => setEditForm((f) => ({ ...f, candidate_1_name: e.target.value }))}
                            style={{ padding: '0.5rem' }}
                        />
                        <select
                            value={editForm.candidate_1_party}
                            onChange={(e) => setEditForm((f) => ({ ...f, candidate_1_party: e.target.value }))}
                            style={{ padding: '0.5rem' }}
                        >
                            <option value="D">D</option>
                            <option value="R">R</option>
                            <option value="I">I</option>
                        </select>
                        <input
                            type="number"
                            placeholder="%"
                            value={editForm.candidate_1_pct}
                            onChange={(e) => setEditForm((f) => ({ ...f, candidate_1_pct: e.target.value }))}
                            style={{ padding: '0.5rem' }}
                        />
                        <label>Candidate 2</label>
                        <input
                            placeholder="Name"
                            value={editForm.candidate_2_name}
                            onChange={(e) => setEditForm((f) => ({ ...f, candidate_2_name: e.target.value }))}
                            style={{ padding: '0.5rem' }}
                        />
                        <select
                            value={editForm.candidate_2_party}
                            onChange={(e) => setEditForm((f) => ({ ...f, candidate_2_party: e.target.value }))}
                            style={{ padding: '0.5rem' }}
                        >
                            <option value="D">D</option>
                            <option value="R">R</option>
                            <option value="I">I</option>
                        </select>
                        <input
                            type="number"
                            placeholder="%"
                            value={editForm.candidate_2_pct}
                            onChange={(e) => setEditForm((f) => ({ ...f, candidate_2_pct: e.target.value }))}
                            style={{ padding: '0.5rem' }}
                        />
                        <label>Margin (positive = candidate 1 leads)</label>
                        <input
                            type="number"
                            step="0.1"
                            placeholder="e.g. 3.5"
                            value={editForm.margin}
                            onChange={(e) => setEditForm((f) => ({ ...f, margin: e.target.value }))}
                            style={{ padding: '0.5rem' }}
                        />
                        <input
                            placeholder="Source note"
                            value={editForm.source_note}
                            onChange={(e) => setEditForm((f) => ({ ...f, source_note: e.target.value }))}
                            style={{ padding: '0.5rem' }}
                        />
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
                        <button type="button" className="election-tab" onClick={saveMapData}>Save</button>
                        <button type="button" className="election-tab" onClick={deleteMapData} style={{ background: '#6b2d2d' }}>Delete</button>
                        <button type="button" className="election-tab" onClick={() => setEditingState(null)}>Cancel</button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ElectionMap;
