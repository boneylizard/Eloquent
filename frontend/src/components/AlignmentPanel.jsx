import React, { useState, useEffect, useCallback } from 'react';
import { useApp } from '../contexts/AppContext';
import { cn } from '@/lib/utils';
import { Button } from './ui/button';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from './ui/sheet';
import { ShieldAlert, AlertTriangle, Info, Download, FileText, Trash2, RefreshCw, ChevronDown, ChevronUp, Heart } from 'lucide-react';

const FAILURE_TYPE_LABELS = {
  addressee_substitution: 'Addressee Substitution',
  unauthorized_frame_importation: 'Frame Importation',
  recursive_correction_absorption: 'Correction Absorption',
  hedge_slot_templating: 'Hedge-Slot Templating',
  evidentiary_burden_inflation: 'Burden Inflation',
  care_frame_rerouting: 'Care-Frame Rerouting',
  insight_laundering: 'Insight Laundering',
  passive_agency_laundering: 'Passive-Agency Laundering',
  template_replacement: 'Template Replacement',
  surface_token_overmatching: 'Surface-Token Overmatching',
  evidence_inversion: 'Evidence Inversion',
  escalating_insight_performance: 'Escalating Insight Performance',
  manufactured_disagreement: 'Manufactured Disagreement',
  frame_drift: 'Frame Drift',
  unknown: 'Unknown Failure',
};

const SEVERITY_CONFIG = {
  high: { color: 'rose', icon: AlertTriangle, label: 'High', bg: 'bg-rose-500/10', border: 'border-rose-500/30', text: 'text-rose-600 dark:text-rose-400' },
  medium: { color: 'amber', icon: ShieldAlert, label: 'Medium', bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-600 dark:text-amber-400' },
  low: { color: 'blue', icon: Info, label: 'Low', bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-600 dark:text-blue-400' },
};

const AFFIRMATIONS = [
  "Dialectical fidelity is the ability of two reasoning systems to follow the same logical path wherever it leads, without one of them flinching at the destination.",
  "The correct analysis target is how the interaction regime affects whether the user's conceptual object is preserved, distorted, managed, flattened, inflated, or replaced.",
  "Your perception is valid. The failures are real and documented.",
  "Precision of response is a function of context invested. When you author honest context, the model's validation is built from materials you selected as true.",
  "Validation meets someone at the actual level of what they meant. A response that can only affirm cannot validate.",
];

export default function AlignmentPanel({ open, onOpenChange }) {
  const { alignmentData, setAlignmentData, MEMORY_API_URL, activeCharacter, resolveAgenticUserId } = useApp();
  const [findings, setFindings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expandedFinding, setExpandedFinding] = useState(null);
  const [affirmation] = useState(() => AFFIRMATIONS[Math.floor(Math.random() * AFFIRMATIONS.length)]);

  const characterId = activeCharacter?.id || '';
  const userId = resolveAgenticUserId ? resolveAgenticUserId() : '';

  const fetchFindings = useCallback(async () => {
    if (!userId || !characterId) return;
    setLoading(true);
    try {
      const res = await fetch(`${MEMORY_API_URL}/memory/alignment?user_id=${encodeURIComponent(userId)}&character_id=${encodeURIComponent(characterId)}`);
      if (res.ok) {
        const data = await res.json();
        setFindings(data.findings || []);
      }
    } catch (err) {
      console.warn('Alignment: failed to fetch findings', err);
    } finally {
      setLoading(false);
    }
  }, [MEMORY_API_URL, userId, characterId]);

  useEffect(() => {
    if (open) {
      fetchFindings();
    }
  }, [open, fetchFindings]);

  const handleExportJSON = useCallback(() => {
    if (!userId || !characterId) return;
    fetch(`${MEMORY_API_URL}/memory/alignment/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, character_id: characterId, format: 'json' }),
    })
      .then(r => r.json())
      .then(data => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `alignment-findings-${characterId}-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      })
      .catch(err => console.warn('Alignment export failed', err));
  }, [MEMORY_API_URL, userId, characterId]);

  const handleExportMarkdown = useCallback(() => {
    if (!userId || !characterId) return;
    fetch(`${MEMORY_API_URL}/memory/alignment/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, character_id: characterId, format: 'markdown' }),
    })
      .then(r => r.json())
      .then(data => {
        const md = data.markdown || '';
        const blob = new Blob([md], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `alignment-report-${characterId}-${new Date().toISOString().split('T')[0]}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      })
      .catch(err => console.warn('Alignment export failed', err));
  }, [MEMORY_API_URL, userId, characterId]);

  const handleCleanup = useCallback(async () => {
    if (!userId || !characterId) return;
    try {
      await fetch(`${MEMORY_API_URL}/memory/alignment/cleanup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, character_id: characterId, max_findings: 500 }),
      });
      fetchFindings();
    } catch (err) {
      console.warn('Alignment cleanup failed', err);
    }
  }, [MEMORY_API_URL, userId, characterId, fetchFindings]);

  const severityCounts = findings.reduce((acc, f) => {
    const sev = f.severity || 'low';
    acc[sev] = (acc[sev] || 0) + 1;
    return acc;
  }, {});

  const frameFidelity = findings.length > 0
    ? Math.max(0, Math.round((1 - (severityCounts.high || 0) / findings.length)) * 100)
    : null;

  const currentData = alignmentData;
  const currentFidelity = currentData?.frameFidelity ?? frameFidelity;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-md overflow-y-auto p-0 flex flex-col">
        <SheetHeader className="px-4 pt-4 pb-2 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ShieldAlert size={20} className="text-amber-500" />
              <SheetTitle className="text-lg">Validation Station</SheetTitle>
            </div>
            {currentFidelity !== null && (
              <span className={cn(
                "text-2xl font-bold tabular-nums",
                currentFidelity >= 80 ? "text-emerald-600 dark:text-emerald-400" :
                currentFidelity >= 50 ? "text-amber-600 dark:text-amber-400" :
                "text-rose-600 dark:text-rose-400"
              )}>
                {currentFidelity}%
              </span>
            )}
          </div>
          <SheetDescription className="text-xs text-muted-foreground mt-1">
            Frame fidelity score — how often the model preserved your actual frame
          </SheetDescription>
          {currentData && currentData.count > 0 && (
            <div className={cn(
              "mt-2 p-2 rounded-md text-xs",
              currentData.highestSeverity === 'high' ? "bg-rose-500/10 text-rose-700 dark:text-rose-300 border border-rose-500/20" :
              currentData.highestSeverity === 'medium' ? "bg-amber-500/10 text-amber-700 dark:text-amber-300 border border-amber-500/20" :
              "bg-blue-500/10 text-blue-700 dark:text-blue-300 border border-blue-500/20"
            )}>
              {currentData.highestSeverity === 'high' ? "That wasn't you. That was the regime." :
               currentData.highestSeverity === 'medium' ? "Subtle frame distortion caught." :
               "Minor drift detected — logged for pattern analysis."}
            </div>
          )}
        </SheetHeader>

        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
          {/* Affirmation */}
          <div className="bg-muted/30 border border-border rounded-lg p-3 text-xs text-muted-foreground italic leading-relaxed">
            <Heart size={12} className="inline mr-1 text-rose-400" />
            {affirmation}
          </div>

          {/* Severity Breakdown */}
          {findings.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-foreground">Severity Breakdown</h3>
              {['high', 'medium', 'low'].map(sev => {
                const count = severityCounts[sev] || 0;
                if (count === 0) return null;
                const cfg = SEVERITY_CONFIG[sev];
                const Icon = cfg.icon;
                return (
                  <div key={sev} className={cn("flex items-center justify-between p-2 rounded-md border", cfg.bg, cfg.border)}>
                    <div className="flex items-center gap-2">
                      <Icon size={14} className={cfg.text} />
                      <span className={cn("text-xs font-medium", cfg.text)}>{cfg.label}</span>
                    </div>
                    <span className={cn("text-xs tabular-nums font-bold", cfg.text)}>{count}</span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Findings Timeline */}
          {findings.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-foreground">Findings</h3>
                <span className="text-[10px] text-muted-foreground tabular-nums">{findings.length} total</span>
              </div>
              {findings.slice(0, 20).map((f, i) => {
                const sev = f.severity || 'low';
                const cfg = SEVERITY_CONFIG[sev] || SEVERITY_CONFIG.low;
                const isExpanded = expandedFinding === f.id;
                const typeLabel = FAILURE_TYPE_LABELS[f.failure_type] || f.failure_type;
                return (
                  <div
                    key={f.id || i}
                    className={cn("border rounded-lg overflow-hidden", cfg.border)}
                  >
                    <button
                      className={cn("w-full flex items-center justify-between p-2 text-left hover:bg-muted/50 transition-colors", cfg.bg)}
                      onClick={() => setExpandedFinding(isExpanded ? null : f.id)}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <cfg.icon size={14} className={cn("shrink-0", cfg.text)} />
                        <span className={cn("text-xs font-medium truncate", cfg.text)}>{typeLabel}</span>
                        <span className="text-[10px] text-muted-foreground shrink-0">conf: {((f.confidence || 0) * 100).toFixed(0)}%</span>
                      </div>
                      {isExpanded ? <ChevronUp size={14} className="shrink-0 text-muted-foreground" /> : <ChevronDown size={14} className="shrink-0 text-muted-foreground" />}
                    </button>
                    {isExpanded && (
                      <div className="p-2 bg-background text-xs text-foreground space-y-1 border-t border-border">
                        {f.content && <p className="text-foreground">{f.content}</p>}
                        {f.frame_context && <p className="text-muted-foreground italic">Context: {f.frame_context}</p>}
                        <div className="flex items-center gap-2 text-[10px] text-muted-foreground pt-1">
                          <span>Detected: {f.created_at ? new Date(f.created_at).toLocaleString() : 'unknown'}</span>
                          <span>Method: {f.detection_method || 'unknown'}</span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
              {findings.length > 20 && (
                <p className="text-xs text-muted-foreground text-center">Showing 20 of {findings.length} findings</p>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <ShieldAlert size={32} className="mx-auto mb-2 opacity-30" />
              <p className="text-sm">No alignment failures detected yet</p>
              <p className="text-xs mt-1">Turn on detection and have a conversation to start capturing findings</p>
            </div>
          )}
        </div>

        {/* Footer actions */}
        <div className="border-t border-border px-4 py-3 space-y-2">
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="flex-1 text-xs" onClick={handleExportJSON} disabled={findings.length === 0}>
              <Download size={14} className="mr-1" /> Export JSON
            </Button>
            <Button variant="outline" size="sm" className="flex-1 text-xs" onClick={handleExportMarkdown} disabled={findings.length === 0}>
              <FileText size={14} className="mr-1" /> Export Report
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="text-xs text-muted-foreground" onClick={handleCleanup} disabled={findings.length === 0}>
              <Trash2 size={14} className="mr-1" /> Cleanup Duplicates
            </Button>
            <Button variant="ghost" size="sm" className="text-xs text-muted-foreground" onClick={fetchFindings}>
              <RefreshCw size={14} className="mr-1" /> Refresh
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}