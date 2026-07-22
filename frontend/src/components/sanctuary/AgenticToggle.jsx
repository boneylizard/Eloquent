/**
 * AgenticToggle — compact switch for Sanctuary agentic mode.
 *
 * Features:
 *  - Switch with ON/OFF states
 *  - Status indicator (idle → connecting → active → error)
 *  - "Test Pipeline" button with LIVE dynamic event display (SSE events render as they arrive)
 *  - Each stage shows: analysis, somatic, cipher, text, done — with timing
 *  - Broadcasts identity to CognitiveGlass standalone window
 */

import React, { useState, useCallback, useRef } from 'react';
import { Switch } from '../ui/switch';
import { Zap, CheckCircle2, AlertCircle, Loader2, Wrench } from 'lucide-react';
import { getBackendUrl } from '../../config/api';
import { demuxAgenticStream } from '../../utils/agenticStream';
import { broadcastToGlass } from '../../utils/sanctuaryWindowSync';
import { openCognitiveGlassWindow } from '../../utils/sanctuaryWindowSync';

const STAGES = ['analysis', 'somatic', 'cipher', 'text', 'done'];

export default function AgenticToggle({
  enabled, onToggle, disabled, characterName, userId, characterId, modelName,
  cognitiveGlassCtx, shadowStateCtx, somaticCtx, interfaceHijackCtx,
}) {
  const [status, setStatus] = useState('idle');
  const [statusMsg, setStatusMsg] = useState('');
  const [liveEvents, setLiveEvents] = useState([]);
  const [isTesting, setIsTesting] = useState(false);
  const [testDone, setTestDone] = useState(null);
  const [accumText, setAccumText] = useState('');
  const testRef = useRef({ stage: null, started: 0 });

  const handleToggle = useCallback(async (checked) => {
    onToggle(checked);
    if (checked) {
      setStatus('connecting');
      setStatusMsg('Checking endpoint...');
      try {
        const res = await fetch(`${getBackendUrl()}/agentic/state/${encodeURIComponent(userId||'test')}/${encodeURIComponent(characterId||'test')}`);
        if (res.ok) { setStatus('active'); setStatusMsg('Ready'); }
        else { throw new Error(`${res.status}`); }
      } catch (e) {
        setStatus('error');
        setStatusMsg(`Endpoint error: ${e.message}`);
      }
    } else {
      setStatus('idle'); setStatusMsg(''); setLiveEvents([]); setTestDone(null); setAccumText('');
    }
  }, [onToggle, userId, characterId]);

  const handleTest = useCallback(async () => {
    if (isTesting || !enabled || !modelName) return;
    setIsTesting(true);
    setLiveEvents([]);
    setTestDone(null);
    setAccumText('');
    testRef.current = { stage: null, started: Date.now() };

    // Reset and prepare
    cognitiveGlassCtx?.reset();
    cognitiveGlassCtx?.startTurn();

    try {
      const testPayload = {
        user_id: userId || 'test_user',
        character_id: characterId || 'test_char',
        conversation_id: `test-${Date.now()}`,
        prompt: `<start_of_turn>system\nYou are Zara, a test assistant. Be direct and concise.\n<end_of_turn>\n<start_of_turn>user\nVerify the agentic pipeline is functional.\n<end_of_turn>\n<start_of_turn>model\n`,
        model_name: modelName,
        max_tokens: 96,
        temperature: 0.7, top_p: 0.9, top_k: 40, repetition_penalty: 1.1,
        stream: true, gpu_id: 0,
        userProfile: { id: userId || 'test_user' },
        history: [{ role: 'user', content: 'Hello' }, { role: 'assistant', content: 'Ready.' }],
        active_character: { name: 'Test Character', description: 'A test' },
      };

      const res = await fetch(`${getBackendUrl()}/agentic/turn`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testPayload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      await demuxAgenticStream(res, {
        onAnalysis: (data) => {
          const evt = { type: 'analysis', data, time: Date.now() };
          setLiveEvents(prev => [...prev, evt]);
          cognitiveGlassCtx?.addAnalysisReasoning(data);
          shadowStateCtx?.applyAnalysis(data);
        },
        onSomatic: (data) => {
          const evt = { type: 'somatic', data, time: Date.now() };
          setLiveEvents(prev => [...prev, evt]);
          somaticCtx?.applySomatic(data);
          cognitiveGlassCtx?.addSomaticLabel(data);
          if (data.interface_hijack) interfaceHijackCtx?.applyHijack(data.interface_hijack, data.python_drive);
        },
        onCipher: (data) => {
          const evt = { type: 'cipher', data, time: Date.now() };
          setLiveEvents(prev => [...prev, evt]);
          shadowStateCtx?.storeCipher(data);
          cognitiveGlassCtx?.addCipherGlyphs(data);
        },
        onText: (delta) => {
          setAccumText(prev => prev + delta);
          setLiveEvents(prev => {
            const last = prev[prev.length - 1];
            if (last && last.type === 'text') {
              return [...prev.slice(0, -1), { ...last, data: last.data + delta, chars: (last.chars || 0) + delta.length }];
            }
            return [...prev, { type: 'text', data: delta, time: Date.now(), chars: delta.length }];
          });
        },
        onDone: (meta) => {
          const evt = { type: 'done', data: meta, time: Date.now() };
          setLiveEvents(prev => [...prev, evt]);
          cognitiveGlassCtx?.endTurn();
          shadowStateCtx?.applyDone(meta);
          setTestDone('pass');
          setStatus('active');
          setStatusMsg('Pipeline verified');
        },
        onError: (err) => {
          setLiveEvents(prev => [...prev, { type: 'error', data: err, time: Date.now() }]);
          setTestDone('fail');
        },
      });
    } catch (e) {
      setLiveEvents(prev => [...prev, { type: 'error', data: { detail: e.message }, time: Date.now() }]);
      setTestDone('fail');
      setStatus('error');
      setStatusMsg(`Test failed: ${e.message}`);
      cognitiveGlassCtx?.endTurn();
    } finally {
      setIsTesting(false);
      if (!testDone) setTestDone('fail');
    }
  }, [isTesting, enabled, modelName, userId, characterId, cognitiveGlassCtx, shadowStateCtx, somaticCtx, interfaceHijackCtx, testDone]);

  const handlePopOutGlass = async () => {
    const w = await openCognitiveGlassWindow();
    if (w) {
      // Send identity once the window loads
      setTimeout(() => {
        broadcastToGlass({
          type: 'identity',
          data: { userId: userId || '', characterId: characterId || '', characterName: characterName || '' },
        }, 'main');
        // Start the agentic turn in the pop-out window
        broadcastToGlass({
          type: 'start_turn',
          ts: Date.now(),
        }, 'main');
      }, 500);
      // Update status to indicate pop-out is active
      setStatus('active');
      setStatusMsg('Cognitive Glass window opened');
    }
  };

  const stageComplete = (stage) => {
    const idx = STAGES.indexOf(stage);
    const seen = new Set(liveEvents.map(e => e.type));
    return STAGES.slice(0, idx).every(s => seen.has(s)) && seen.has(stage);
  };

  return (
    <div style={{
      background: 'rgba(15,15,30,0.85)',
      border: '1px solid rgba(100,150,255,0.12)',
      borderRadius: '0.4rem',
      padding: '0.4rem 0.6rem',
      fontFamily: "'Courier New', monospace",
      maxWidth: '340px',
    }}>
      {/* Switch row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Zap size={14} style={{ color: enabled ? 'rgba(120,180,255,0.9)' : 'rgba(120,140,200,0.3)' }} />
          <span style={{ fontSize: '0.65rem', color: enabled ? 'rgba(180,210,255,0.9)' : 'rgba(120,140,200,0.4)' }}>
            {enabled ? 'Agentic ON' : 'Agentic OFF'}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          {status === 'connecting' && <Loader2 size={10} className="animate-spin" style={{ color: 'rgba(120,160,255,0.7)' }} />}
          {status === 'active' && <CheckCircle2 size={10} style={{ color: 'rgba(100,220,150,0.8)' }} />}
          {status === 'error' && <AlertCircle size={10} style={{ color: 'rgba(255,130,100,0.8)' }} />}
          <Switch checked={enabled} onCheckedChange={handleToggle} disabled={disabled} />
        </div>
      </div>

      {/* Status line */}
      {statusMsg && (
        <div style={{ fontSize: '0.55rem', marginTop: '0.25rem', color: status === 'error' ? 'rgba(255,130,100,0.7)' : status === 'active' ? 'rgba(100,220,150,0.6)' : 'rgba(120,160,255,0.5)' }}>
          {statusMsg}
        </div>
      )}

      {/* Actions when enabled */}
      {enabled && (
        <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.35rem' }}>
          <button onClick={handleTest} disabled={isTesting || !modelName}
            style={{
              display: 'flex', alignItems: 'center', gap: '0.25rem', padding: '0.2rem 0.4rem',
              fontSize: '0.55rem', fontFamily: 'inherit',
              background: 'rgba(100,150,255,0.08)', color: 'rgba(150,180,255,0.6)',
              border: '1px solid rgba(100,150,255,0.15)', borderRadius: '0.2rem',
              cursor: isTesting ? 'not-allowed' : 'pointer', opacity: isTesting ? 0.5 : 1,
            }}>
            {isTesting ? <><Loader2 size={9} className="animate-spin" /> Testing...</> : <><Wrench size={9} /> Test Pipeline</>}
          </button>
          <button onClick={handlePopOutGlass}
            style={{
              display: 'flex', alignItems: 'center', gap: '0.25rem', padding: '0.2rem 0.4rem',
              fontSize: '0.55rem', fontFamily: 'inherit',
              background: 'transparent', color: 'rgba(120,140,200,0.4)',
              border: '1px solid rgba(100,150,255,0.08)', borderRadius: '0.2rem',
              cursor: 'pointer',
            }}>
            ↗ Pop Out Glass
          </button>
        </div>
      )}

      {/* LIVE event stream — renders events in real-time as they arrive */}
      {(liveEvents.length > 0) && (
        <div style={{
          marginTop: '0.35rem', borderTop: '1px solid rgba(100,150,255,0.06)', paddingTop: '0.3rem',
          maxHeight: '160px', overflowY: 'auto',
        }}>
          {/* Stage progress bar */}
          <div style={{ display: 'flex', gap: '0.15rem', marginBottom: '0.3rem' }}>
            {STAGES.map(s => {
              const done = stageComplete(s);
              const isCurrent = liveEvents.length > 0 && liveEvents[liveEvents.length - 1]?.type === s;
              return (
                <div key={s} style={{
                  flex: 1, height: '3px', borderRadius: '2px',
                  background: done ? 'rgba(100,220,150,0.6)' : isCurrent ? 'rgba(120,160,255,0.4)' : 'rgba(100,120,160,0.1)',
                  transition: 'background 0.2s',
                }} title={`${s} ${done ? '✓' : isCurrent ? '>' : '·'}`} />
              );
            })}
          </div>

          {/* Events */}
          {liveEvents.map((evt, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'flex-start', gap: '0.3rem', fontSize: '0.5rem',
              padding: '0.08rem 0', color: evt.type === 'error' ? 'rgba(255,130,100,0.7)' : 'rgba(150,180,220,0.5)',
            }}>
              <span style={{
                minWidth: '2.8rem', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.04em',
                color: evt.type === 'analysis' ? 'rgba(120,200,255,0.7)' :
                       evt.type === 'somatic' ? 'rgba(255,120,200,0.7)' :
                       evt.type === 'cipher' ? 'rgba(200,150,255,0.7)' :
                       evt.type === 'text' ? 'rgba(120,255,180,0.7)' :
                       evt.type === 'done' ? 'rgba(255,220,120,0.7)' : 'rgba(255,120,100,0.7)',
              }}>{evt.type}</span>
              <span style={{ color: 'rgba(120,140,200,0.3)' }}>:{(evt.time - testRef.current.started)}ms</span>
              {evt.type === 'text' && (
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '150px' }}>
                  {evt.data?.slice(-40)}
                </span>
              )}
              {evt.type === 'done' && <span style={{ color: 'rgba(100,220,150,0.5)' }}>✓ {evt.data?.final_state?.posture}</span>}
              {evt.type === 'error' && <span style={{ color: 'rgba(255,130,100,0.6)' }}>{evt.data?.detail?.slice(0, 40)}</span>}
            </div>
          ))}

          {/* Accumulated text preview */}
          {accumText && (
            <div style={{ marginTop: '0.25rem', padding: '0.2rem 0.3rem', background: 'rgba(20,40,30,0.15)', borderRadius: '0.15rem', fontSize: '0.5rem', color: 'rgba(150,220,180,0.4)', maxHeight: '40px', overflowY: 'auto' }}>
              {accumText.length > 120 ? accumText.slice(-120) : accumText}
            </div>
          )}

          {/* Test result banner */}
          {testDone && (
            <div style={{ marginTop: '0.25rem', fontSize: '0.55rem', color: testDone === 'pass' ? 'rgba(100,220,150,0.7)' : 'rgba(255,130,100,0.7)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              {testDone === 'pass' ? <><CheckCircle2 size={10} /> Pipeline verified — all {liveEvents.length} events</> : <><AlertCircle size={10} /> Test failed</>}
            </div>
          )}
        </div>
      )}

      {/* Enhanced visual feedback for pipeline stages */}
      {enabled && liveEvents.length > 0 && (
        <div style={{ marginTop: '0.3rem', padding: '0.2rem', background: 'rgba(60,100,180,0.1)', borderRadius: '0.2rem', border: '1px solid rgba(100,150,255,0.08)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <span style={{ fontSize: '0.45rem', color: 'rgba(120,180,255,0.8)', fontWeight: 600 }}>
              Agentic Pipeline Status
            </span>
            <span style={{ fontSize: '0.4rem', color: 'rgba(150,180,255,0.5)' }}>
              {liveEvents.filter(e => e.type === 'done').length}/{STAGES.length} stages complete
            </span>
          </div>
          
          {/* Visual stage indicators */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.1rem' }}>
            {STAGES.map((stage, idx) => {
              const isCompleted = stageComplete(stage);
              const isActive = liveEvents.length > 0 && liveEvents[liveEvents.length - 1]?.type === stage;
              const isPending = !isCompleted && !isActive;
              
              return (
                <div key={stage} style={{
                  height: '6px', 
                  borderRadius: '3px',
                  background: isCompleted ? 'rgba(100,220,150,0.8)' : isActive ? 'rgba(120,160,255,0.6)' : 'rgba(100,120,160,0.2)',
                  transition: 'all 0.3s ease',
                  boxShadow: isActive ? '0 0 8px rgba(120,160,255,0.4)' : 'none',
                }} title={`${stage} ${isCompleted ? '✓ Complete' : isActive ? '→ Active' : '○ Pending'}`} />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
