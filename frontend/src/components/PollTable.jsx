import { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { getBackendUrl } from '../config/api';
import './ElectionTracker.css';

const PollTable = ({ polls, metadata = {}, raceType }) => {
    const { PRIMARY_API_URL } = useApp();
    const [sortConfig, setSortConfig] = useState({ key: 'date_added', direction: 'desc' });
    const [candidateRoster, setCandidateRoster] = useState([]);

    useEffect(() => {
        const baseUrl = PRIMARY_API_URL || getBackendUrl();
        fetch(`${baseUrl}/election/candidates`)
            .then((r) => r.ok ? r.json() : { candidates: [] })
            .then((data) => setCandidateRoster(data.candidates || []))
            .catch(() => setCandidateRoster([]));
    }, [PRIMARY_API_URL]);

    if (!polls || polls.length === 0) {
        return (
            <div className="poll-table-container">
                <div className="loading-message">
                    <p>Loading polling data... This may take 15-20 seconds.</p>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                        We're scraping live data from multiple sources.
                    </p>
                </div>
            </div>
        );
    }

    // Extract race context from metadata
    const raceTitle = metadata.title || '';
    const hasFavorability = metadata.has_favorability || false;
    const availableRaces = metadata.races || [];
    const headers = metadata.headers || [];
    const isInfogramTable = headers.some(h => /race or candidate/i.test(h)) || polls.some(p => p.race);
    const hasPrimaryRaces = polls.some(p => /(primary|runoff)/i.test(String(p.race || '')));
    const hasAnyState = polls.some(p => p.state);

    // Keep server order (already sorted by date, newest first)
    const sortedPolls = [...polls];

    const requestSort = (key) => {
        let direction = 'ascending';
        if (sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    const getSortIndicator = (name) => {
        if (sortConfig.key !== name) return '';
        return sortConfig.direction === 'ascending' ? ' ▲' : ' ▼';
    };

    return (
        <div className="poll-table-container">
            {(raceTitle || raceType === 'house') && (
                <div style={{
                    padding: '1rem',
                    background: 'var(--bg-tertiary)',
                    borderBottom: '1px solid var(--border-color)',
                    fontWeight: 'bold',
                    fontSize: '1.1rem'
                }}>
                    {raceTitle || '2026 House – Individual Races'}
                    {availableRaces.length > 1 && (
                        <div style={{ fontSize: '0.85rem', fontWeight: 'normal', marginTop: '0.5rem', color: 'var(--text-secondary)' }}>
                            Showing {availableRaces.length} races combined
                        </div>
                    )}
                </div>
            )}

            {isInfogramTable ? (
                <table className="poll-table poll-table--infogram">
                    <thead>
                        <tr>
                            <th>#</th>
                            {hasAnyState && <th>State</th>}
                            <th>Added</th>
                            <th>Race or Candidate</th>
                            <th>Pollster</th>
                            <th>{raceType === 'approval' ? 'Net' : 'Lead'}</th>
                            <th>{raceType === 'approval' ? 'Approve' : hasPrimaryRaces ? 'Candidate 1' : 'Dem or Fav'}</th>
                            <th>{raceType === 'approval' ? 'Disapprove' : hasPrimaryRaces ? 'Candidate 2' : 'GOP or Unfav'}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedPolls.map((poll, index) => {
                            const addedRaw = poll.added || poll.date_added || poll.date_range || '';
                            const added = formatPollDate(poll.date_iso, addedRaw);
                            const race = poll.race || poll.candidate || '-';
                            const pollster = poll.pollster || '-';
                            const isApproval = raceType === 'approval';
                            const lead = isApproval
                                ? (getApprovalNet(poll) || '-')
                                : (poll.lead || poll.margin || getLeadFromResults(poll) || '-');
                            const isPrimary = /(primary|runoff)/i.test(String(race));
                            const isDemPrimary = /(democratic|dem)\s+primary|(democratic|dem)\s+runoff/i.test(String(race));
                            const isRepPrimary = /(republican|gop)\s+primary|(republican|gop)\s+runoff/i.test(String(race));
                            const [cell1, cell2] = getResultPairForDisplay(poll, isPrimary);
                            const { badge, text: raceText } = splitRace(race);
                            const leadClass = getLeadClassForRow(lead, isDemPrimary, isRepPrimary, raceType, candidateRoster, poll.state, poll.results);
                            const { chip1Class, chip2Class } = getChipClassesForRow(lead, isDemPrimary, isRepPrimary, cell1, cell2, raceType, candidateRoster, poll.state, poll.results);
                            const stateDisplay = poll.state || (badge && badge.length <= 3 ? badge : '') || '-';

                            return (
                                <tr key={index} className="poll-row">
                                    <td className="poll-cell poll-cell--index">
                                        <span className="poll-index">{poll.row_num || index + 1}</span>
                                    </td>
                                    {hasAnyState && (
                                        <td className="poll-cell poll-cell--state">
                                            <span className="poll-state">{stateDisplay}</span>
                                        </td>
                                    )}
                                    <td className="poll-cell poll-cell--added">
                                        <span className="poll-added">{added}</span>
                                    </td>
                                    <td className="poll-cell poll-cell--race">
                                        <span className="poll-race-text">{race}</span>
                                    </td>
                                    <td>
                                        {poll.link ? (
                                            <a href={poll.link} target="_blank" rel="noopener noreferrer" className="pollster-link">
                                                {pollster}
                                            </a>
                                        ) : (
                                            <span className="pollster-text">{pollster}</span>
                                        )}
                                    </td>
                                    <td className="poll-cell poll-cell--lead">
                                        <span className={`poll-lead ${leadClass}`}>{lead}</span>
                                    </td>
                                    <td className="poll-cell poll-cell--dem">
                                        <span className={`poll-chip ${chip1Class}`}>
                                            {isPrimary && cell1.label ? `${cell1.label} ${cell1.value}` : (cell1.value ?? '-')}
                                        </span>
                                    </td>
                                    <td className="poll-cell poll-cell--gop">
                                        <span className={`poll-chip ${chip2Class}`}>
                                            {isPrimary && cell2.label ? `${cell2.label} ${cell2.value}` : (cell2.value ?? '-')}
                                        </span>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            ) : (
                <table className="poll-table">
                    <thead>
                        <tr>
                            <th onClick={() => requestSort('pollster')}>
                                Pollster{getSortIndicator('pollster')}
                            </th>
                            <th onClick={() => requestSort('date_range')}>
                                Date{getSortIndicator('date_range')}
                            </th>
                            <th onClick={() => requestSort('sample')}>
                                Sample{getSortIndicator('sample')}
                            </th>
                            <th>Grade</th>
                            <th>Results</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedPolls.map((poll, index) => (
                            <tr key={index}>
                                <td>
                                    {poll.link ? (
                                        <a href={poll.link} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-primary)' }}>
                                            {poll.pollster}
                                        </a>
                                    ) : (
                                        poll.pollster
                                    )}
                                </td>
                                <td>{poll.date_range}</td>
                                <td>{poll.sample}</td>
                                <td>
                                    <span className={`poll-grade ${getGradeClass(poll.grade)}`}>
                                        {poll.grade || '-'}
                                    </span>
                                </td>
                                <td>
                                    {formatResults(poll)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
};

function getGradeClass(grade) {
    if (!grade) return '';
    const g = grade.toUpperCase();
    if (g.startsWith('A')) return 'grade-A';
    if (g.startsWith('B')) return 'grade-B';
    if (g.startsWith('C')) return 'grade-C';
    if (g.startsWith('D')) return 'grade-D';
    return 'grade-F';
}

function formatResults(poll) {
    // Approval rating: net is approve - disapprove (no D/R, no blue/red)
    if (poll.approve && poll.disapprove) {
        const net = getApprovalNet(poll);
        return (
            <span style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Approve {poll.approve}</span>
                <span style={{ color: 'var(--text-secondary)' }}>Disapprove {poll.disapprove}</span>
                <span className="poll-lead--neutral" style={{ fontWeight: 'bold' }}>
                    {net ?? poll.margin ?? '-'}
                </span>
            </span>
        );
    }

    // Generic ballot or simple D vs R
    if (poll.dem && poll.gop) {
        const isDemLead = poll.margin && poll.margin.includes('D');
        return (
            <span style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <span className="text-dem">D {poll.dem}</span>
                <span className="text-gop">R {poll.gop}</span>
                <span className={isDemLead ? 'text-dem' : 'text-gop'} style={{ fontWeight: 'bold' }}>
                    {poll.margin}
                </span>
            </span>
        );
    }

    // Individual candidate races (Senate, House, Governor)
    // Results come as object mapping candidate names to percentages
    if (poll.results && typeof poll.results === 'object' && !Array.isArray(poll.results)) {
        const entries = Object.entries(poll.results);

        if (entries.length === 0) {
            // No candidates extracted - show raw data
            return <span style={{ color: 'orange' }}>No candidate data found</span>;
        }

        return (
            <span style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
                {entries.map(([candidate, percentage], idx) => (
                    <span key={idx} style={{ fontSize: '0.9rem' }}>
                        <strong>{candidate}:</strong> {percentage}
                    </span>
                ))}
                {poll.margin && (
                    <span style={{ fontWeight: 'bold', color: 'var(--accent-primary)', marginLeft: '0.5rem' }}>
                        {poll.margin}
                    </span>
                )}
            </span>
        );
    }

    // Fallback for array results
    if (poll.results && Array.isArray(poll.results)) {
        return (
            <span style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                {poll.results.map((result, idx) => (
                    <span key={idx} style={{
                        padding: '0.25rem 0.5rem',
                        background: 'var(--bg-tertiary)',
                        borderRadius: '4px',
                        fontSize: '0.85rem'
                    }}>
                        {result}
                    </span>
                ))}
            </span>
        );
    }

    // Fallback - show margin if available
    if (poll.margin) {
        return <span style={{ fontWeight: 'bold' }}>{poll.margin}</span>;
    }

    return '-';
}

function toDisplayValue(v) {
    if (v == null || v === '') return '-';
    return String(v).trim() || '-';
}

function getResultPair(poll) {
    if (poll.results && typeof poll.results === 'object' && !Array.isArray(poll.results)) {
        const values = Object.values(poll.results).map(v => (v != null && v !== '' ? String(v).trim() : null)).filter(Boolean);
        if (values.length >= 2) return [values[0], values[1]];
        if (values.length === 1) return [values[0], '-'];
    }
    if (poll.dem && poll.gop) return [toDisplayValue(poll.dem), toDisplayValue(poll.gop)];
    if (poll.approve && poll.disapprove) return [toDisplayValue(poll.approve), toDisplayValue(poll.disapprove)];
    if (poll.dem_or_fav || poll.gop_or_unfav) return [toDisplayValue(poll.dem_or_fav), toDisplayValue(poll.gop_or_unfav)];
    return ['-', '-'];
}

/** True if a results key looks like a race/category (e.g. "Texas Dem Primary") rather than a candidate name. */
function isRaceLikeKey(key, pollRace) {
    if (!key || typeof key !== 'string') return true;
    const k = key.trim();
    if (k.length > 40) return true;
    if (pollRace && k === String(pollRace).trim()) return true;
    if (/primary|runoff|gop primary|dem primary|republican primary|democratic primary/i.test(k)) return true;
    return false;
}

/** For display: returns [{ label, value }, { label, value }]. For primaries, label is candidate name. */
function getResultPairForDisplay(poll, isPrimary) {
    const empty = { label: '', value: '-' };
    if (isPrimary && poll.results && typeof poll.results === 'object' && !Array.isArray(poll.results)) {
        const entries = Object.entries(poll.results).filter(([, v]) => v != null && v !== '');
        if (entries.length >= 2) {
            if (!isRaceLikeKey(entries[0][0], poll.race) && !isRaceLikeKey(entries[1][0], poll.race))
                return [{ label: entries[0][0], value: entries[0][1] }, { label: entries[1][0], value: entries[1][1] }];
        }
        if (entries.length >= 1 && !isRaceLikeKey(entries[0][0], poll.race))
            return [{ label: entries[0][0], value: entries[0][1] }, empty];
        if (entries.length >= 1 && isRaceLikeKey(entries[0][0], poll.race)) {
            const [a, b] = getResultPair(poll);
            return [{ label: '', value: a }, { label: '', value: b }];
        }
    }
    const [a, b] = getResultPair(poll);
    return [{ label: '', value: a }, { label: '', value: b }];
}

function getLeadFromResults(poll) {
    if (!poll.results || typeof poll.results !== 'object') return '';
    const values = Object.values(poll.results);
    for (const v of values) {
        if (typeof v === 'string' && (v.includes('+') || v.toLowerCase().includes('tie'))) {
            return v;
        }
    }
    return '';
}

function splitRace(race) {
    if (!race || typeof race !== 'string') return { badge: '', text: '-' };
    const cleaned = race.trim();
    if (cleaned.includes(' - ')) {
        const [left, right] = cleaned.split(' - ', 2);
        const badge = left.length <= 6 ? left : '';
        return { badge, text: right || cleaned };
    }
    return { badge: '', text: cleaned };
}

/** Format poll date with year so it's clear (e.g. "Dec 22, 2025") — avoids confusion that "December 22" could be 2026. */
function formatPollDate(dateIso, fallback) {
    if (!dateIso || typeof dateIso !== 'string') return fallback || '-';
    const m = dateIso.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (!m) return fallback || dateIso;
    const [, y, mo, d] = m;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = months[parseInt(mo, 10) - 1] || mo;
    return `${month} ${parseInt(d, 10)}, ${y}`;
}

/** Do not treat name+num as party: "Klobuchar+22" must not match 'r+'. */
function getLeadClass(lead) {
    if (!lead || lead === '-') return 'poll-lead--muted';
    const lower = lead.toLowerCase();
    if (lower.includes('(d)') || lower.includes('democrat')) return 'poll-lead--dem';
    if (lower.includes('(r)') || lower.includes('republican') || lower.includes('gop')) return 'poll-lead--gop';
    if (/^d\s*\+/i.test(lead.trim()) || /\s+d\s*\+\s*-?[\d.]+/.test(lower)) return 'poll-lead--dem';
    if (/^r\s*\+/i.test(lead.trim()) || /\s+r\s*\+\s*-?[\d.]+/.test(lower)) return 'poll-lead--gop';
    if (lower.includes('fav') || lower.includes('unfav')) return 'poll-lead--neutral';
    return 'poll-lead--neutral';
}

/** Extract leader name from lead string, e.g. "Rogers+3" -> "Rogers", "Craig+7" -> "Craig". */
function getLeaderNameFromLead(lead) {
    if (!lead || typeof lead !== 'string') return null;
    const trimmed = lead.trim();
    const match = trimmed.match(/^(.+?)\s*[+-]\s*-?[\d.]+/);
    return match ? match[1].trim() : null;
}

/** Lead column: approval = neutral; primaries = party color; else getLeadClass; if still neutral, use roster to resolve leader name -> party; if leader not in roster, infer from other candidate's party. */
function getLeadClassForRow(lead, isDemPrimary, isRepPrimary, raceType, roster, state, results) {
    if (raceType === 'approval') return 'poll-lead--neutral';
    if (isDemPrimary) return 'poll-lead--dem';
    if (isRepPrimary) return 'poll-lead--gop';
    const fromText = getLeadClass(lead);
    if (fromText !== 'poll-lead--neutral') return fromText;
    if (!roster || !roster.length) return 'poll-lead--neutral';
    const leaderName = getLeaderNameFromLead(lead);
    if (!leaderName) return 'poll-lead--neutral';
    const office = { senate: 'senate', governor: 'governor', house: 'house' }[raceType];
    let party = getPartyFromRoster(leaderName, roster, state, office);
    if (party === 'D' || party === 'G' || party === 'L') return 'poll-lead--dem';
    if (party === 'R') return 'poll-lead--gop';
    // Leader not in roster: infer from the other candidate (e.g. "Acton+1" vs Ramaswamy -> Ramaswamy is R so Acton = D)
    if (results && typeof results === 'object' && !Array.isArray(results)) {
        const names = Object.keys(results);
        const norm = (s) => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
        const leaderNorm = norm(leaderName);
        const otherName = names.find((n) => norm(n) !== leaderNorm);
        if (otherName) {
            const otherParty = getPartyFromRoster(otherName, roster, state, office);
            if (otherParty === 'R') return 'poll-lead--dem';
            if (otherParty === 'D' || otherParty === 'G' || otherParty === 'L') return 'poll-lead--gop';
        }
    }
    return 'poll-lead--neutral';
}

/** Parse lead string for leader party: 'Husted (R) +3' -> 'R', 'Democrats +4' -> 'D'. Do not treat name+num as party (e.g. Klobuchar+22 must not match 'r+'). */
function getLeaderPartyFromLead(lead) {
    if (!lead || typeof lead !== 'string') return null;
    const lower = lead.toLowerCase();
    if (lower.includes('(r)') || lower.includes('republican')) return 'R';
    if (lower.includes('(d)') || lower.includes('democrat')) return 'D';
    // Only treat R+ / D+ as party when they are the lead prefix (e.g. "R+5"), not inside a name (e.g. "Klobuchar+22")
    if (/^r\s*\+/i.test(lead.trim()) || /\s+r\s*\+\s*-?\d/.test(lower)) return 'R';
    if (/^d\s*\+/i.test(lead.trim()) || /\s+d\s*\+\s*-?\d/.test(lower)) return 'D';
    return null;
}

/** Look up party for a name in the roster; state and office narrow the match. */
function getPartyFromRoster(name, roster, state, office) {
    if (!name || !roster || !roster.length) return null;
    const norm = (s) => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
    const nName = norm(String(name));
    if (!nName) return null;
    const o = (office || '').toLowerCase();
    const s = (state || '').toUpperCase();
    for (const c of roster) {
        const cState = (c.state || '').toUpperCase();
        const cOffice = (c.office || '').toLowerCase();
        if (s && cState !== s) continue;
        if (o && cOffice !== o) continue;
        if (norm(c.name) === nName) return (c.party || '').toUpperCase();
        for (const a of c.aliases || []) {
            if (norm(a) === nName) return (c.party || '').toUpperCase();
        }
    }
    return null;
}

/** Chip columns: use lead (R)/(D) when available; else use candidate roster; else col1=blue, col2=red. */
function getChipClassesForRow(lead, isDemPrimary, isRepPrimary, cell1, cell2, raceType, roster, state, results) {
    const v1 = cell1?.value;
    const v2 = cell2?.value;
    const muted1 = !v1 && v1 !== 0 ? ' poll-chip--muted' : '';
    const muted2 = !v2 && v2 !== 0 ? ' poll-chip--muted' : '';
    if (raceType === 'approval') return { chip1Class: 'poll-chip--muted', chip2Class: 'poll-chip--muted' };
    if (isDemPrimary) return { chip1Class: `poll-chip--dem${muted1}`, chip2Class: `poll-chip--dem${muted2}` };
    if (isRepPrimary) return { chip1Class: `poll-chip--gop${muted1}`, chip2Class: `poll-chip--gop${muted2}` };
    const leaderParty = getLeaderPartyFromLead(lead);
    const num1 = parseFloat(String(v1).replace(/%/g, '').trim());
    const num2 = parseFloat(String(v2).replace(/%/g, '').trim());
    const v1IsLeader = !Number.isNaN(num1) && !Number.isNaN(num2) && num1 >= num2;
    if (leaderParty === 'R') {
        return {
            chip1Class: v1IsLeader ? `poll-chip--gop${muted1}` : `poll-chip--dem${muted1}`,
            chip2Class: v1IsLeader ? `poll-chip--dem${muted2}` : `poll-chip--gop${muted2}`,
        };
    }
    if (leaderParty === 'D') {
        return {
            chip1Class: v1IsLeader ? `poll-chip--dem${muted1}` : `poll-chip--gop${muted1}`,
            chip2Class: v1IsLeader ? `poll-chip--gop${muted2}` : `poll-chip--dem${muted2}`,
        };
    }
    const office = { senate: 'senate', governor: 'governor', house: 'house' }[raceType];
    const name1 = cell1?.label || (results && typeof results === 'object' && !Array.isArray(results) ? Object.keys(results)[0] : null);
    const name2 = cell2?.label || (results && typeof results === 'object' && !Array.isArray(results) ? Object.keys(results)[1] : null);
    if (roster && roster.length && (name1 || name2)) {
        const p1 = getPartyFromRoster(name1, roster, state, office);
        const p2 = getPartyFromRoster(name2, roster, state, office);
        const dem = (p) => p === 'D' || p === 'G' || p === 'L';
        const gop = (p) => p === 'R';
        const chip = (p, muted) => (p && dem(p)) ? `poll-chip--dem${muted}` : (p && gop(p)) ? `poll-chip--gop${muted}` : 'poll-chip--muted';
        const oppositeChip = (p, muted) => (p && gop(p)) ? `poll-chip--dem${muted}` : (p && dem(p)) ? `poll-chip--gop${muted}` : 'poll-chip--muted';
        if (p1 && p2) {
            return { chip1Class: chip(p1, muted1), chip2Class: chip(p2, muted2) };
        }
        if (p1) {
            return { chip1Class: chip(p1, muted1), chip2Class: oppositeChip(p1, muted2) };
        }
        if (p2) {
            return { chip1Class: oppositeChip(p2, muted1), chip2Class: chip(p2, muted2) };
        }
        // Fallback: resolve leader from lead string (e.g. "Klobuchar+22") and assign leader = higher value
        const leaderName = getLeaderNameFromLead(lead);
        const leaderParty = leaderName ? getPartyFromRoster(leaderName, roster, state, office) : null;
        if (leaderParty && (dem(leaderParty) || gop(leaderParty))) {
            const leaderChip = chip(leaderParty, '');
            const otherChip = oppositeChip(leaderParty, '');
            return {
                chip1Class: v1IsLeader ? leaderChip + muted1 : otherChip + muted1,
                chip2Class: v1IsLeader ? otherChip + muted2 : leaderChip + muted2,
            };
        }
    }
    return {
        chip1Class: 'poll-chip--muted',
        chip2Class: 'poll-chip--muted',
    };
}

/** Approval net: approve - disapprove. Returns e.g. "-4" or "+3" (no D/R). Uses approve/disapprove or dem_or_fav/gop_or_unfav or results. */
function getApprovalNet(poll) {
    let a = poll.approve ?? poll.results?.Approve ?? poll.results?.approve ?? poll.dem_or_fav;
    let d = poll.disapprove ?? poll.results?.Disapprove ?? poll.results?.disapprove ?? poll.gop_or_unfav;
    a = parseFloat(String(a ?? '').replace(/%/g, '').trim());
    d = parseFloat(String(d ?? '').replace(/%/g, '').trim());
    if (Number.isNaN(a) || Number.isNaN(d)) return null;
    const net = Math.round((a - d) * 10) / 10;
    if (net === 0) return '0.0';
    const str = net > 0 ? `+${net}` : `${net}`;
    return str.includes('.') ? str : `${str}.0`;
}

export default PollTable;
