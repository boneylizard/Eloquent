/**
 * AgenticStatusPanel — displays the current agentic pipeline status
 * and provides enhanced visual feedback for the UX.
 *
 * Features:
 *   - Shows current pipeline stage with visual indicators
 *   - Displays shadow state metrics (heat, dominance, posture)
 *   - Provides real-time progress feedback
 *   - Enhanced visual presentation with better colors and animations
 */

import React, { useMemo } from 'react';
import { Brain, Activity, Zap, Thermometer, TrendingUp, Shield, Eye, Wind, Heart, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useCognitiveGlass } from '../../contexts/CognitiveGlassContext';
import { useShadowState } from '../../contexts/ShadowStateContext';
import { useSomaticDashboard } from '../../contexts/SomaticDashboardContext';

export default function AgenticStatusPanel() {
  const {
    reasoningEntries,
    cipherGlyphs,
    sceneFrames,
    isStreaming,
  } = useCognitiveGlass();
  const { shadowState } = useShadowState();
  const { somaticPayload } = useSomaticDashboard();

  // Get current pipeline stage
  const currentStage = useMemo(() => {
    if (!reasoningEntries.length && !cipherGlyphs.length && !sceneFrames.length) return null;
    
    // Determine current stage based on what we have
    if (reasoningEntries.length > 0) return 'analysis';
    if (cipherGlyphs.length > 0) return 'cipher';
    if (sceneFrames.length > 0) return 'scene';
    return 'somatic';
  }, [reasoningEntries, cipherGlyphs, sceneFrames]);

  // Get stage configuration
  const getStageConfig = (stage) => {
    const configs = {
      analysis: {
        icon: Brain,
        label: 'Analysis',
        color: 'rgba(120, 200, 255, 0.9)',
        bgColor: 'rgba(120, 200, 255, 0.1)',
        description: 'Evaluating context and determining posture'
      },
      somatic: {
        icon: Activity,
        label: 'Somatic',
        color: 'rgba(255, 120, 200, 0.9)',
        bgColor: 'rgba(255, 120, 200, 0.1)',
        description: 'Generating physical and emotional state'
      },
      cipher: {
        icon: Zap,
        label: 'Cipher',
        color: 'rgba(200, 150, 255, 0.9)',
        bgColor: 'rgba(200, 150, 255, 0.1)',
        description: 'Encoding shadow state for next turn'
      },
      scene: {
        icon: Shield,
        label: 'Scene',
        color: 'rgba(100, 220, 150, 0.9)',
        bgColor: 'rgba(100, 220, 150, 0.1)',
        description: 'Rendering scene frame changes'
      },
      text: {
        icon: Heart,
        label: 'Text',
        color: 'rgba(255, 220, 120, 0.9)',
        bgColor: 'rgba(255, 220, 120, 0.1)',
        description: 'Generating natural language response'
      },
      done: {
        icon: CheckCircle2,
        label: 'Complete',
        color: 'rgba(100, 220, 150, 0.9)',
        bgColor: 'rgba(100, 220, 150, 0.1)',
        description: 'Turn completed successfully'
      }
    };
    return configs[stage] || configs.analysis;
  };

  // Get shadow state metrics
  const getShadowStateMetrics = () => {
    return [
      {
        label: 'Heat Index',
        value: shadowState.heat_index?.toFixed(2) || '0.00',
        icon: Thermometer,
        color: 'rgba(255, 130, 100, 0.8)',
        unit: ''
      },
      {
        label: 'Dominance',
        value: shadowState.dominance_vector?.toFixed(2) || '0.50',
        icon: TrendingUp,
        color: 'rgba(120, 180, 255, 0.8)',
        unit: ''
      },
      {
        label: 'Posture',
        value: shadowState.posture || 'neutral',
        icon: Shield,
        color: 'rgba(100, 220, 150, 0.8)',
        unit: ''
      },
      {
        label: 'Trap Progress',
        value: shadowState.trap_progress?.toFixed(2) || '0.00',
        icon: AlertCircle,
        color: 'rgba(255, 120, 200, 0.8)',
        unit: ''
      }
    ];
  };

  // Get somatic indicators
  const getSomaticIndicators = () => {
    if (!somaticPayload?.dashboard) return [];
    
    const indicators = [];
    const dashboard = somaticPayload.dashboard;
    
    if (dashboard.lubrication_level) {
      indicators.push({
        label: 'Lubrication',
        value: dashboard.lubrication_level,
        icon: Heart,
        color: 'rgba(255, 120, 200, 0.8)'
      });
    }
    
    if (dashboard.pupil_dilation) {
      indicators.push({
        label: 'Pupils',
        value: `${Math.round(dashboard.pupil_dilation * 100)}%`,
        icon: Eye,
        color: 'rgba(120, 200, 255, 0.8)'
      });
    }
    
    if (dashboard.spatial_position) {
      indicators.push({
        label: 'Position',
        value: dashboard.spatial_position,
        icon: Shield,
        color: 'rgba(100, 220, 150, 0.8)'
      });
    }
    
    if (dashboard.breath_rate) {
      indicators.push({
        label: 'Breath',
        value: dashboard.breath_rate,
        icon: Wind,
        color: 'rgba(255, 220, 120, 0.8)'
      });
    }
    
    return indicators;
  };

  const shadowMetrics = getShadowStateMetrics();
  const somaticIndicators = getSomaticIndicators();
  const currentStageConfig = currentStage ? getStageConfig(currentStage) : null;

  return (
    <div style={{
      background: 'rgba(15, 15, 30, 0.9)',
      border: '1px solid rgba(100, 150, 255, 0.2)',
      borderRadius: '0.5rem',
      padding: '1rem',
      fontFamily: "'Courier New', monospace",
      maxWidth: '400px',
      margin: '0 auto',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '1rem',
        paddingBottom: '0.5rem',
        borderBottom: '1px solid rgba(100, 150, 255, 0.1)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Brain size={16} style={{ color: 'rgba(120, 180, 255, 0.9)' }} />
          <span style={{ fontSize: '0.9rem', color: 'rgba(180, 210, 255, 0.9)', fontWeight: 600 }}>
            Agentic Pipeline Status
          </span>
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.3rem',
          padding: '0.2rem 0.5rem',
          borderRadius: '1rem',
          background: isStreaming ? 'rgba(100, 220, 150, 0.1)' : 'rgba(100, 120, 160, 0.1)',
          border: `1px solid ${isStreaming ? 'rgba(100, 220, 150, 0.3)' : 'rgba(100, 120, 160, 0.3)'}`,
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: isStreaming ? 'rgba(100, 220, 150, 0.9)' : 'rgba(100, 120, 160, 0.5)',
            animation: isStreaming ? 'pulse 1.5s infinite' : 'none',
          }} />
          <span style={{ fontSize: '0.4rem', color: isStreaming ? 'rgba(100, 220, 150, 0.8)' : 'rgba(100, 120, 160, 0.6)' }}>
            {isStreaming ? 'LIVE' : 'IDLE'}
          </span>
        </div>
      </div>

      {/* Current Stage Indicator */}
      {currentStageConfig && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.8rem',
          padding: '0.8rem',
          background: currentStageConfig.bgColor,
          borderRadius: '0.3rem',
          marginBottom: '1rem',
          border: `1px solid ${currentStageConfig.color}20`,
        }}>
          <currentStageConfig.icon size={20} style={{ color: currentStageConfig.color }} />
          <div>
            <div style={{ fontSize: '0.7rem', color: currentStageConfig.color, fontWeight: 600 }}>
              {currentStageConfig.label}
            </div>
            <div style={{ fontSize: '0.45rem', color: 'rgba(150, 180, 255, 0.7)' }}>
              {currentStageConfig.description}
            </div>
          </div>
        </div>
      )}

      {/* Shadow State Metrics */}
      {shadowMetrics.length > 0 && (
        <div>
          <div style={{
            fontSize: '0.6rem',
            color: 'rgba(120, 180, 255, 0.8)',
            fontWeight: 600,
            marginBottom: '0.5rem',
          }}>
            Shadow State
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '0.5rem',
          }}>
            {shadowMetrics.map((metric, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.5rem',
                  background: 'rgba(20, 40, 60, 0.3)',
                  borderRadius: '0.2rem',
                  border: `1px solid ${metric.color}20`,
                }}
              >
                <metric.icon size={12} style={{ color: metric.color }} />
                <div>
                  <div style={{ fontSize: '0.45rem', color: metric.color, fontWeight: 600 }}>
                    {metric.label}
                  </div>
                  <div style={{ fontSize: '0.4rem', color: 'rgba(150, 180, 255, 0.6)' }}>
                    {metric.value}{metric.unit}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Somatic Indicators */}
      {somaticIndicators.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <div style={{
            fontSize: '0.6rem',
            color: 'rgba(255, 120, 200, 0.8)',
            fontWeight: 600,
            marginBottom: '0.5rem',
          }}>
            Somatic State
          </div>
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '0.3rem',
          }}>
            {somaticIndicators.map((indicator, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  padding: '0.3rem 0.5rem',
                  background: 'rgba(60, 20, 40, 0.3)',
                  borderRadius: '1rem',
                  border: `1px solid ${indicator.color}30`,
                }}
              >
                <indicator.icon size={10} style={{ color: indicator.color }} />
                <span style={{ fontSize: '0.35rem', color: indicator.color, fontWeight: 600 }}>
                  {indicator.label}: {indicator.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pipeline Progress */}
      <div style={{ marginTop: '1rem' }}>
        <div style={{
          fontSize: '0.6rem',
          color: 'rgba(100, 220, 150, 0.8)',
          fontWeight: 600,
          marginBottom: '0.5rem',
        }}>
          Pipeline Progress
        </div>
        <div style={{
          display: 'flex',
          gap: '0.2rem',
          height: '4px',
        }}>
          {['analysis', 'somatic', 'cipher', 'text', 'done'].map((stage, idx) => {
            const isCompleted = currentStage && ['analysis', 'somatic', 'cipher', 'text', 'done'].indexOf(stage) < ['analysis', 'somatic', 'cipher', 'text', 'done'].indexOf(currentStage);
            const isActive = currentStage === stage;
            return (
              <div
                key={stage}
                style={{
                  flex: 1,
                  height: '4px',
                  borderRadius: '2px',
                  background: isCompleted ? 'rgba(100, 220, 150, 0.8)' : isActive ? 'rgba(120, 160, 255, 0.6)' : 'rgba(100, 120, 160, 0.2)',
                  transition: 'background 0.3s ease',
                  boxShadow: isActive ? '0 0 6px rgba(120, 160, 255, 0.4)' : 'none',
                }}
                title={`${stage} ${isCompleted ? '✓' : isActive ? '→' : '○'}`}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
