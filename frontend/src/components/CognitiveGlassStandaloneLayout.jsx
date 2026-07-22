/**
 * CognitiveGlassStandaloneLayout — standalone popup window for the CognitiveGlass.
 *
 * Renders the full CognitiveGlass with its own provider stack, subscribing to
 * BroadcastChannel messages from the main window for two-way bidirectional sync.
 * Changes here (reset baseline, etc.) broadcast back to the main window.
 */

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Brain, Zap, Eye, Droplets, Maximize2, Minimize2, Activity, Thermometer, ShieldAlert, RefreshCw, WifiOff, User, Sparkles, ChevronDown, ChevronRight, GripVertical, Hand, MessageSquare, AlertTriangle } from 'lucide-react';
import { ShadowStateProvider, useShadowState } from '../contexts/ShadowStateContext';
import { SomaticDashboardProvider, useSomaticDashboard } from '../contexts/SomaticDashboardContext';
import { CognitiveGlassProvider, useCognitiveGlass } from '../contexts/CognitiveGlassContext';
import { InterfaceHijackProvider, useInterfaceHijack } from '../contexts/InterfaceHijackContext';
import { AgenticProfileProvider, useAgenticProfile } from '../contexts/AgenticProfileContext';
import CipherGlyphRow from './sanctuary/CipherGlyphRow';
import SceneFrameRenderer from './sanctuary/SceneFrameRenderer';
import ThoughtLeakTerminal from './sanctuary/ThoughtLeakTerminal';
import TypewriterText from './sanctuary/TypewriterText';
import AgenticStatusPanel from './sanctuary/AgenticStatusPanel';
import { subscribeSanctuarySync, broadcastToMain } from '../utils/sanctuaryWindowSync';

export default function CognitiveGlassStandaloneLayout() {
  return (
      <AgenticProfileProvider>
      <ShadowStateProvider>
        <SomaticDashboardProvider>
          <CognitiveGlassProvider>
            <InterfaceHijackProvider>
              <StandaloneContent />
            </InterfaceHijackProvider>
          </CognitiveGlassProvider>
        </SomaticDashboardProvider>
      </ShadowStateProvider>
      </AgenticProfileProvider>
  );
}

function decodeCipherBlock(block) {
  if (!block) return null;
  const CIPHER_RE = /⟦CIPHER:v(\d+):([0-9a-fA-F]+):([A-Za-z0-9+/=]*)⟧/;
  const match = CIPHER_RE.exec(block);
  if (!match) return null;

  const [, versionStr, hexLayer, b64Layer] = match;
  const version = parseInt(versionStr, 10);
  if (isNaN(version)) return null;

  let turnCount = 0;
  try {
    turnCount = parseInt(hexLayer.slice(4, 12), 16);
    if (isNaN(turnCount)) turnCount = 0;
  } catch { turnCount = 0; }

  let heatIndex = 0.0;
  try {
    const hexPart = hexLayer.slice(12);
    if (hexPart.length >= 4) {
      heatIndex = parseInt(hexPart.slice(0, 4), 16) / 0xFFFF;
    }
  } catch { heatIndex = 0.0; }

  let remainder = {};
  try {
    const decoded = atob(b64Layer);
    remainder = JSON.parse(decoded);
  } catch { remainder = {}; }

  return {
    version,
    turn_count: turnCount,
    heat_index: heatIndex,
    ...remainder,
  };
}

function SanctuaryEntry({ entry, index, showDebug }) {
  const [isExpanded, setIsExpanded] = useState(true);
  
  return (
    <div 
      className={`sanctuary-glass-entry ${entry.type}`}
      style={{ animationDelay: `${index * 0.05}s` }}
    >
      <div className="sanctuary-glass-entry-header">
        <div className="sanctuary-glass-entry-header-left">
          <button 
            className="sanctuary-glass-expand-btn"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          {entry.type === 'analysis' && (
            <span className="sanctuary-glass-entry-type">ANALYSIS</span>
          )}
          {entry.type === 'somatic' && (
            <span className="sanctuary-glass-entry-type somatic">SOMATIC</span>
          )}
          {entry.type === 'tactile' && (
            <span className="sanctuary-glass-entry-type tactile">{entry.labelDisplay || 'TACTILE OUTREACH'}</span>
          )}
          {entry.type === 'signal' && (
            <span className="sanctuary-glass-entry-type signal">{entry.labelDisplay || 'CHARACTER SIGNAL'}</span>
          )}
          <span className="sanctuary-glass-entry-meta">
            {entry.type === 'analysis' && (
              <>
                {entry.posture} · DOM {((entry.dominance ?? 0) * 100).toFixed(0)}%
              </>
            )}
            {entry.type === 'somatic' && (
              <>
                {entry.postureLabel}
                {entry.ghostActive && <span className="sanctuary-glass-tag ghost">GHOST</span>}
                {entry.whoreMode && <span className="sanctuary-glass-tag mode">MODE</span>}
              </>
            )}
            {entry.type === 'tactile' && entry.proximity && (
              <span className="sanctuary-glass-entry-meta-dim">{entry.proximity}</span>
            )}
            {entry.type === 'signal' && entry.voiceThisTurn && (
              <span className="sanctuary-glass-entry-meta-dim">"{entry.voiceThisTurn}"</span>
            )}
          </span>
        </div>
      </div>
      
      {entry.type === 'analysis' && entry.reasoning && entry.reasoning.trim() && (
        <div className={`sanctuary-glass-entry-body ${isExpanded ? 'expanded' : 'collapsed'}`}>
          {entry.reasoning}
        </div>
      )}
      
      {entry.type === 'analysis' && entry.emotional_state && entry.emotional_state.trim() && (
        <div className={`sanctuary-glass-entry-detail ${isExpanded ? 'expanded' : 'collapsed'}`}>
          <span className="sanctuary-glass-detail-label">Emotional:</span>
          {entry.emotional_state}
        </div>
      )}
      
      {entry.type === 'analysis' && entry.physical_state && entry.physical_state.trim() && (
        <div className={`sanctuary-glass-entry-detail ${isExpanded ? 'expanded' : 'collapsed'}`}>
          <span className="sanctuary-glass-detail-label">Physical:</span>
          {entry.physical_state}
        </div>
      )}
      
      {entry.type === 'analysis' && entry.trajectory && entry.trajectory.trim() && (
        <div className={`sanctuary-glass-entry-detail ${isExpanded ? 'expanded' : 'collapsed'}`}>
          <span className="sanctuary-glass-detail-label">Trajectory:</span>
          {entry.trajectory}
        </div>
      )}

      {entry.type === 'analysis' && entry.internal_state && entry.internal_state.trim() && (
        <div className={`sanctuary-glass-entry-narrative ${isExpanded ? 'expanded' : 'collapsed'}`}>
          <span className="sanctuary-glass-narrative-label">INTERNAL STATE:</span>
          {entry.internal_state}
        </div>
      )}

      {entry.type === 'analysis' && entry.external_state && entry.external_state.trim() && (
        <div className={`sanctuary-glass-entry-narrative external ${isExpanded ? 'expanded' : 'collapsed'}`}>
          <span className="sanctuary-glass-narrative-label">EXTERNAL STATE:</span>
          {entry.external_state}
        </div>
      )}

      {entry.type === 'somatic' && entry.somatic_narrative && entry.somatic_narrative.trim() && (
        <div className={`sanctuary-glass-entry-narrative somatic ${isExpanded ? 'expanded' : 'collapsed'}`}>
          <span className="sanctuary-glass-narrative-label">SOMATIC NARRATIVE:</span>
          {entry.somatic_narrative}
        </div>
      )}

      {entry.type === 'somatic' && entry.behavioral_cues && entry.behavioral_cues.trim() && (
        <div className={`sanctuary-glass-entry-narrative behavioral ${isExpanded ? 'expanded' : 'collapsed'}`}>
          <span className="sanctuary-glass-narrative-label">BEHAVIORAL CUES:</span>
          {entry.behavioral_cues}
        </div>
      )}

      {entry.type === 'tactile' && (
        <>
          {entry.pose && entry.pose.trim() && (
            <div className={`sanctuary-glass-entry-narrative tactile-pose ${isExpanded ? 'expanded' : 'collapsed'}`}>
              <span className="sanctuary-glass-narrative-label">POSE:</span>
              {entry.pose}
            </div>
          )}
          {entry.gestureNarrative && entry.gestureNarrative.trim() && (
            <div className={`sanctuary-glass-entry-narrative tactile-gesture ${isExpanded ? 'expanded' : 'collapsed'}`}>
              <span className="sanctuary-glass-narrative-label">GESTURE:</span>
              {entry.gestureNarrative}
            </div>
          )}
          {entry.proximity && entry.proximity.trim() && (
            <div className={`sanctuary-glass-entry-detail ${isExpanded ? 'expanded' : 'collapsed'}`}>
              <span className="sanctuary-glass-detail-label">Proximity:</span>
              {entry.proximity}
            </div>
          )}
        </>
      )}

      {entry.type === 'signal' && (
        <>
          {entry.covertAction && entry.covertAction.trim() && (
            <div className={`sanctuary-glass-entry-narrative signal-covert ${isExpanded ? 'expanded' : 'collapsed'}`}>
              <span className="sanctuary-glass-narrative-label">COVERT ACTION:</span>
              {entry.covertAction}
            </div>
          )}
          {entry.voiceThisTurn && entry.voiceThisTurn.trim() && (
            <div className={`sanctuary-glass-entry-detail ${isExpanded ? 'expanded' : 'collapsed'}`}>
              <span className="sanctuary-glass-detail-label">Voice This Turn:</span>
              {entry.voiceThisTurn}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function StandaloneContent() {
  const {
    reasoningEntries,
    cipherGlyphs,
    sceneFrames,
    isStreaming,
    startTurn,
    addAnalysisReasoning,
    addSomaticLabel,
    addTactileEntry,
    addSignalEntry,
    addCipherGlyphs,
    endTurn,
    reset: resetGlass,
  } = useCognitiveGlass();
  const {
    shadowState,
    applyAnalysis,
    storeCipher,
    applyDone,
    resetBaseline,
    isResetting,
  } = useShadowState();
  const { somaticPayload, applySomatic, reset: resetSomatic } = useSomaticDashboard();
  const { hijackState, applyHijack, reset: resetHijack } = useInterfaceHijack();
  const { displayConfig } = useAgenticProfile();

  const gaugeConfig = displayConfig?.gauges || {
    heat: { label: 'Heat Index', color: 'rgba(255, 130, 100, 0.8)' },
    dominance: { label: 'Dominance', color: 'rgba(120, 180, 255, 0.8)' },
    trap: { label: 'Trap Progress', color: 'rgba(255, 120, 200, 0.8)' },
  };

  const somaticDashboard = somaticPayload?.dashboard || {};
  const hasSomaticData = somaticDashboard?.lubrication_level || 
                        somaticDashboard?.pupil_dilation || 
                        somaticDashboard?.spatial_position || 
                        somaticDashboard?.breath_rate || 
                        somaticPayload?.posture_label;

  const [connected, setConnected] = useState(false);
  const [userId, setUserId] = useState('');
  const [characterId, setCharacterId] = useState('');
  const [characterName, setCharacterName] = useState('');
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [showDebug, setShowDebug] = useState(false);
  const [showStatus, setShowStatus] = useState(false);
  const [pipelineEvents, setPipelineEvents] = useState([]);
  const reconnectTimeoutRef = useRef(null);
  const subscribedRef = useRef(false);

  const heatIndex = shadowState?.heat_index || 0;
  const dominanceVector = shadowState?.dominance_vector || 0.5;
  const trapProgress = shadowState?.trap_progress || 0;

  const scrollSpeed = Math.max(3, 15 - heatIndex * 10 - (dominanceVector - 0.5) * 5);

  const bgHue = 220 + (heatIndex * 60);
  const bgSaturation = 30 + (heatIndex * 40);
  const bgLightness = 8 + (heatIndex * 8);
  const accentColor = heatIndex > 0.7 ? 'rgba(255, 80, 150, 0.9)' : 
                      heatIndex > 0.4 ? 'rgba(150, 100, 255, 0.8)' : 
                      'rgba(100, 180, 255, 0.8)';

  const attemptReconnect = useCallback(() => {
    if (reconnectAttempt >= 3) return;
    setReconnectAttempt(prev => prev + 1);
    broadcastToMain({ type: 'glass_ready', ts: Date.now() }, 'main');
    broadcastToMain({ type: 'start_turn', ts: Date.now() }, 'main');
  }, [reconnectAttempt]);

  useEffect(() => {
    if (subscribedRef.current) return;
    subscribedRef.current = true;

    setConnected(false);

    const unsub = subscribeSanctuarySync((msg) => {
      if (msg.direction !== 'main->glass') return;
      
      setConnected(true);
      setReconnectAttempt(0);

      if (showDebug) {
        setPipelineEvents(prev => [...prev, { 
          type: msg.type, 
          ts: Date.now(),
          data: msg.data 
        }]);
      }

      switch (msg.type) {
        case 'start_turn':
          startTurn();
          break;
        case 'analysis':
          addAnalysisReasoning(msg.data);
          applyAnalysis(msg.data);
          break;
        case 'somatic':
          addSomaticLabel(msg.data);
          applySomatic(msg.data);
          if (msg.data.interface_hijack) {
            applyHijack(msg.data.interface_hijack, msg.data.python_drive);
          }
          break;
        case 'tactile':
          addTactileEntry(msg.data);
          break;
        case 'signal':
          addSignalEntry(msg.data);
          break;
        case 'cipher':
          addCipherGlyphs(msg.data);
          storeCipher(msg.data);
          break;
        case 'text':
          break;
        case 'hijack':
          applyHijack(msg.data, null);
          break;
        case 'shadow_state':
          break;
        case 'end_turn':
          endTurn();
          break;
        case 'done':
          applyDone(msg.data);
          endTurn();
          break;
        case 'identity':
          setUserId(msg.data.userId || '');
          setCharacterId(msg.data.characterId || '');
          setCharacterName(msg.data.characterName || '');
          break;
        case 'baseline_reset':
          resetGlass();
          resetSomatic();
          resetHijack();
          break;
      }
    }, 'main');

    broadcastToMain({ type: 'glass_ready', ts: Date.now() }, 'main');

    return () => {
      unsub();
      subscribedRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  const handleResetBaseline = useCallback(async () => {
    if (!userId || !characterId) return;
    const confirmed = window.confirm(`Reset shadow state for ${characterName || 'character'}?\n\nThis clears all tracking back to baseline.`);
    if (!confirmed) return;
    broadcastToMain({ type: 'reset_baseline', data: { userId, characterId } }, 'main');
  }, [userId, characterId, characterName]);

  const sortedEntries = React.useMemo(
    () => [...reasoningEntries].sort((a, b) => a.id - b.id),
    [reasoningEntries]
  );

  const sortedCiphers = React.useMemo(
    () => [...cipherGlyphs].sort((a, b) => a.timestamp - b.timestamp),
    [cipherGlyphs]
  );

  return (
    <div className="sanctuary-glass-alive" style={{
      '--sg-accent': accentColor,
      '--sg-heat': heatIndex,
      '--sg-dominance': dominanceVector,
    }}>
      <div className="sanctuary-glass-bg-mesh" />
      <div className="sanctuary-glass-particles" />
      <ThoughtLeakTerminal events={pipelineEvents} />

      <div className="sanctuary-glass-header-wrap">
        <div className="sanctuary-glass-code-leak">
          <div className="sanctuary-glass-code-leak-inner" style={{ '--sg-scroll-speed': `${scrollSpeed}s` }}>
            <div>01001 11010 10101 :: 0x3A1F 00110 11010 01001 ﾊﾐﾋｰ :;.?! 10010 11010 01001 0xBEEF</div>
            <div>11001 10110 01101 01011 ﾜﾂｵﾘ :: 0xFF 00110 11010 01001 ﾅﾓﾆｻ 10010 11010 01001</div>
            <div>01001 11010 10101 00110 :: 0xDEAD 11010 01001 ﾒｴｶｷ 10010 0xC0DE 11010 01001 11010</div>
            <div>10010 11010 01001 11010 10101 01010 00101 11001 ﾈｽﾀﾇﾍ 10110 01101 01011 00110 ::</div>
            <div>ﾊﾐﾋｰ :: 0xFF 01001 11010 10101 00110 ﾜﾂｵﾘ :: 0x7F 11010 01001 ﾒｴｶｷ 10010 ::</div>
            <div>11010 01001 10010 ﾈｽﾀﾇﾍ :;.?! 11010 01001 11010 10101 01010 :: 0x3A1F 00101 11001</div>
            <div>ﾊﾐﾋｰ :: 0xDEAD 01101 01011 00110 11010 ﾜﾂｵﾘ :: 0xBEEF 01001 10010 11010 01001</div>
            <div>ﾈｽﾀﾇﾍ :: 0xFF 10101 01010 00101 11001 ﾅﾓﾆｻ 10110 01101 01011 00110 11010 01001</div>
            <div>11010 01001 11010 10101 01010 00101 11001 ﾒｴｶｷ 10110 01101 01011 00110 11010 01001</div>
            <div>ﾊﾐﾋｰ :: 0xC0DE 01001 11010 10101 00110 ﾜﾂｵﾘ :: 0x7F 11010 01001 ﾒｴｶｷ 10010 ::</div>
            <div>11010 01001 10010 ﾈｽﾀﾇﾍ :;.?! 11010 01001 11010 10101 01010 :: 0x3A1F 00101 11001</div>
            <div>ﾊﾐﾋｰ :: 0xDEAD 01101 01011 00110 11010 ﾜﾂｵﾘ :: 0xBEEF 01001 10010 11010 01001</div>
            <div>01001 11010 10101 ﾈｽﾀﾇﾍ :: 0xFF 00110 11010 01001 ﾅﾓﾆｻ 10010 11010 01001 ::</div>
            <div>11001 10110 01101 01011 ﾊﾐﾋｰ :: 0x3A1F 00110 11010 01001 ﾜﾂｵﾘ :: 0xBEEF 10010</div>
            <div>10101 01010 00101 11001 10110 01101 01011 ﾒｴｶｷ :: 0x7F 00110 11010 01001 ﾈｽﾀﾇﾍ</div>
            <div>01001 11010 10101 :: 0x3A1F 00110 11010 01001 ﾊﾐﾋｰ :;.?! 10010 11010 01001 0xBEEF</div>
            <div>11001 10110 01101 01011 ﾜﾂｵﾘ :: 0xFF 00110 11010 01001 ﾅﾓﾆｻ 10010 11010 01001</div>
            <div>01001 11010 10101 00110 :: 0xDEAD 11010 01001 ﾒｴｶｷ 10010 0xC0DE 11010 01001 11010</div>
            <div>10010 11010 01001 11010 10101 01010 00101 11001 ﾈｽﾀﾇﾍ 10110 01101 01011 00110 ::</div>
            <div>ﾊﾐﾋｰ :: 0xFF 01001 11010 10101 00110 ﾜﾂｵﾘ :: 0x7F 11010 01001 ﾒｴｶｷ 10010 ::</div>
            <div>11010 01001 10010 ﾈｽﾀﾇﾍ :;.?! 11010 01001 11010 10101 01010 :: 0x3A1F 00101 11001</div>
            <div>ﾊﾐﾋｰ :: 0xDEAD 01101 01011 00110 11010 ﾜﾂｵﾘ :: 0xBEEF 01001 10010 11010 01001</div>
            <div>ﾈｽﾀﾇﾍ :: 0xFF 10101 01010 00101 11001 ﾅﾓﾆｻ 10110 01101 01011 00110 11010 01001</div>
            <div>11010 01001 11010 10101 01010 00101 11001 ﾒｴｶｷ 10110 01101 01011 00110 11010 01001</div>
            <div>ﾊﾐﾋｰ :: 0xC0DE 01001 11010 10101 00110 ﾜﾂｵﾘ :: 0x7F 11010 01001 ﾒｴｶｷ 10010 ::</div>
            <div>11010 01001 10010 ﾈｽﾀﾇﾍ :;.?! 11010 01001 11010 10101 01010 :: 0x3A1F 00101 11001</div>
            <div>ﾊﾐﾋｰ :: 0xDEAD 01101 01011 00110 11010 ﾜﾂｵﾘ :: 0xBEEF 01001 10010 11010 01001</div>
            <div>01001 11010 10101 ﾈｽﾀﾇﾍ :: 0xFF 00110 11010 01001 ﾅﾓﾆｻ 10010 11010 01001 ::</div>
            <div>11001 10110 01101 01011 ﾊﾐﾋｰ :: 0x3A1F 00110 11010 01001 ﾜﾂｵﾘ :: 0xBEEF 10010</div>
            <div>10101 01010 00101 11001 10110 01101 01011 ﾒｴｶｷ :: 0x7F 00110 11010 01001 ﾈｽﾀﾇﾍ</div>
          </div>
        </div>
        <div className="sanctuary-glass-header-alive" style={{ position: 'relative', zIndex: 10 }}>
          <div className="sanctuary-glass-header-left">
            <div className="sanctuary-glass-avatar">
              {characterName ? characterName[0].toUpperCase() : '?'}
            </div>
              <div className="sanctuary-glass-header-info">
                <div className="sanctuary-glass-title">{characterName || 'Unknown'}</div>
                <div className="sanctuary-glass-subtitle">
                  <span className={`sanctuary-glass-status-dot ${connected ? 'connected' : 'disconnected'}`} />
                  {connected ? 'SYNCED' : 'IDLE'}
                </div>
              </div>
          </div>
        <div className="sanctuary-glass-header-right">
          <button 
            onClick={() => setShowStatus(!showStatus)}
            className="sanctuary-glass-debug-btn"
            title="Toggle status panel"
          >
            <span className="sanctuary-glass-debug-text">STATUS</span>
          </button>
          <button 
            onClick={() => setShowDebug(!showDebug)}
            className="sanctuary-glass-debug-btn"
            title="Toggle debug panel"
          >
            <span className="sanctuary-glass-debug-text">DEBUG</span>
          </button>
          {isStreaming && (
            <div className="sanctuary-glass-live-indicator">
              <Activity size={14} className="sanctuary-glass-live-icon" />
              <span>LIVE</span>
            </div>
          )}
          <div className={`sanctuary-glass-status-indicator ${connected ? 'synced' : 'idle'}`}>
            {connected ? 'SYNCED' : 'IDLE'}
          </div>
          <button 
            onClick={handleResetBaseline} 
            disabled={isResetting || isStreaming}
            className="sanctuary-glass-reset-btn"
            title="Reset shadow state to baseline"
          >
            <RefreshCw size={14} />
          </button>
        </div>
        </div>

        <div className="sanctuary-glass-gauges" style={{ position: 'relative', zIndex: 10 }}>
        {(() => {
          const gaugeValues = { heat: heatIndex, dominance: dominanceVector, trap: trapProgress };
          const gaugeIcons = { heat: Thermometer, dominance: ShieldAlert, trap: Eye };
          return Object.entries(gaugeConfig).map(([key, cfg]) => {
            const val = gaugeValues[key] ?? 0;
            const Icon = gaugeIcons[key] || Thermometer;
            return (
              <div className="sanctuary-glass-gauge" key={key}>
                <div className="sanctuary-glass-gauge-label">
                  <Icon size={10} />
                  <span>{cfg?.label || key}</span>
                </div>
                <div className="sanctuary-glass-gauge-bar">
                  <div className="sanctuary-glass-gauge-fill" style={{ width: `${val * 100}%`, background: cfg?.color }} />
                </div>
                <div className="sanctuary-glass-gauge-value">{(val * 100).toFixed(0)}%</div>
              </div>
            );
          });
        })()}
      </div>
      </div>

      <div className="sanctuary-glass-body-alive">
        {sortedEntries.length === 0 && sortedCiphers.length === 0 && sceneFrames.length === 0 && !hasSomaticData && (
          <div className="sanctuary-glass-empty">
            {connected ? (
              <>
                <div className="sanctuary-glass-empty-icon"><Sparkles size={20} /></div>
                <div>Synced. Awaiting neural stream...</div>
              </>
            ) : reconnectAttempt > 0 ? (
              <>
                <div className="sanctuary-glass-empty-icon"><RefreshCw size={20} className="animate-spin" /></div>
                <div>Reconnecting... ({reconnectAttempt}/3)</div>
              </>
            ) : (
              <>
                <div className="sanctuary-glass-empty-icon"><Brain size={20} /></div>
                <div>Enable Agentic mode to activate</div>
              </>
            )}
          </div>
        )}

        {showStatus && (
          <div className="sanctuary-glass-status-panel-wrap" style={{ margin: '0 0.75rem 0.75rem' }}>
            <AgenticStatusPanel />
          </div>
        )}

        {hasSomaticData && (
          <div className="sanctuary-glass-somatic-strip">
            <div className="sanctuary-glass-somatic-label">SOMATIC STATE</div>
            <div className="sanctuary-glass-somatic-chips">
              {somaticDashboard.lubrication_level && (
                <div className="sanctuary-glass-somatic-chip lubrication">
                  <Droplets size={12} />
                  <TypewriterText text={somaticDashboard.lubrication_level} speed={20} enabled={true} />
                </div>
              )}
              {somaticDashboard.pupil_dilation && (
                <div className="sanctuary-glass-somatic-chip pupils">
                  <div 
                    className="sanctuary-glass-pupil-circle" 
                    style={{ 
                      width: `${8 + somaticDashboard.pupil_dilation * 12}px`,
                      height: `${8 + somaticDashboard.pupil_dilation * 12}px`,
                    }} 
                  />
                  <TypewriterText text={`${Math.round(somaticDashboard.pupil_dilation * 100)}%`} speed={20} enabled={true} />
                </div>
              )}
              {somaticDashboard.spatial_position && (
                <div className="sanctuary-glass-somatic-chip position">
                  <Maximize2 size={12} />
                  <TypewriterText text={somaticDashboard.spatial_position} speed={20} enabled={true} />
                </div>
              )}
              {somaticDashboard.breath_rate && (
                <div className={`sanctuary-glass-somatic-chip breath ${somaticDashboard.breath_rate}`}>
                  <Activity size={12} />
                  <TypewriterText text={somaticDashboard.breath_rate} speed={20} enabled={true} />
                </div>
              )}
              {somaticPayload?.posture_label && (
                <div className="sanctuary-glass-somatic-chip posture">
                  <TypewriterText text={somaticPayload.posture_label} speed={20} enabled={true} />
                </div>
              )}
            </div>
          </div>
        )}

        {sortedCiphers.map((c, i) => {
          const decoded = c.block ? decodeCipherBlock(c.block) : null;
          if (!c.glyphs) return null;
          return (
            <div key={`cipher-${i}`} className="sanctuary-glass-cipher-wrapper">
              <div className="sanctuary-glass-cipher-row">
                <div className="sanctuary-glass-cipher-scanline" />
                <CipherGlyphRow glyphs={c.glyphs} block={c.block} phase={c.phase} />
                {decoded?.turn_count !== undefined && (
                  <span className="sanctuary-glass-cipher-turn">TURN {decoded.turn_count}</span>
                )}
              </div>
            </div>
          );
        })}

        {showDebug && (
          <div className="sanctuary-glass-debug-panel system-override">
            <div className="sanctuary-glass-debug-header">
              <span>⚠ SYSTEM OVERRIDE — PIPELINE EVENTS</span>
              <button onClick={() => setPipelineEvents([])} className="sanctuary-glass-debug-clear">CLEAR</button>
            </div>
            {pipelineEvents.map((evt, i) => (
              <div key={i} className="sanctuary-glass-debug-event">
                <span className="sanctuary-glass-debug-type">{evt.type}</span>
                <span className="sanctuary-glass-debug-time">{new Date(evt.ts).toLocaleTimeString()}</span>
                {evt.data && (
                  <pre className="sanctuary-glass-debug-data">{JSON.stringify(evt.data, null, 2)}</pre>
                )}
              </div>
            ))}
          </div>
        )}

        {sceneFrames.map((sf, i) => (
          <SceneFrameRenderer key={`scene-${i}`} xmlFrame={sf.xml} />
        ))}

        {sortedEntries.map((entry, index) => (
          <SanctuaryEntry key={entry.id} entry={entry} index={index} />
        ))}
      </div>
    </div>
  );
}
