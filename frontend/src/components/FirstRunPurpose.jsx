import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  BookOpenText,
  Bot,
  ChevronDown,
  Code2,
  MessagesSquare,
  Mic2,
  Sparkles,
} from 'lucide-react';
import {
  FIRST_RUN_PURPOSES,
  readFirstRunIntent,
  writeFirstRunIntent,
} from '../utils/firstRunIntent';
import {
  INTERFACE_ZOOM_DEFAULT,
  INTERFACE_ZOOM_MAX,
  INTERFACE_ZOOM_MIN,
  readInterfaceZoom,
  setInterfaceZoom,
} from '../utils/interfaceZoom';
import './FirstRunPurpose.css';

const ICONS = {
  roleplay: Sparkles,
  conversation: MessagesSquare,
  writing: BookOpenText,
  'voice-media': Mic2,
  developer: Code2,
  everything: Bot,
};

export default function FirstRunPurpose({ onBegin }) {
  const [purpose, setPurpose] = useState(() => readFirstRunIntent()?.purpose || '');
  const [zoom, setZoom] = useState(readInterfaceZoom);
  const [advanced, setAdvanced] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');

  const adjustZoom = async (next) => {
    const applied = await setInterfaceZoom(next);
    setZoom(applied);
  };

  const begin = async () => {
    if (!purpose || starting) return;
    setStarting(true);
    setError('');
    try {
      const intent = writeFirstRunIntent({ purpose, interfaceZoom: zoom });
      await invoke('begin_runtime_setup');
      onBegin?.(intent);
    } catch (reason) {
      setError(reason?.message || String(reason));
      setStarting(false);
    }
  };

  return (
    <main className="mirid-first-run" aria-labelledby="mirid-first-run-title">
      <section className="mirid-first-run__panel">
        <div className="mirid-first-run__eyebrow">Mirid · first run</div>
        <header className="mirid-first-run__header">
          <h1 id="mirid-first-run-title">What will you mostly use Mirid for?</h1>
          <p>Choose a starting point. Mirid will arrange the first few steps around it; every tool remains available.</p>
        </header>

        <div className="mirid-first-run__choices" role="radiogroup" aria-label="Primary use">
          {FIRST_RUN_PURPOSES.map((item) => {
            const Icon = ICONS[item.id] || Bot;
            const selected = purpose === item.id;
            return (
              <button
                key={item.id}
                type="button"
                role="radio"
                aria-checked={selected}
                className={`mirid-first-run__choice${selected ? ' is-selected' : ''}${item.id === 'roleplay' ? ' is-roleplay' : ''}`}
                onClick={() => setPurpose(item.id)}
              >
                <Icon aria-hidden="true" />
                <span>
                  <strong>{item.label}</strong>
                  <small>{item.description}</small>
                </span>
              </button>
            );
          })}
        </div>

        {purpose === 'roleplay' && (
          <div className="mirid-first-run__roleplay-note">
            <Sparkles aria-hidden="true" />
            <p><strong>A character-first welcome.</strong> After model setup, Mirid will open a guided character room in its Faraday-inspired theme.</p>
          </div>
        )}

        <button
          type="button"
          className="mirid-first-run__advanced-toggle"
          aria-expanded={advanced}
          onClick={() => setAdvanced((current) => !current)}
        >
          <ChevronDown aria-hidden="true" className={advanced ? 'is-open' : ''} />
          Advanced first-run settings
        </button>

        {advanced && (
          <div className="mirid-first-run__advanced">
            <div>
              <strong>Interface size</strong>
              <p>Set this before the runtime download. You can change it later with Ctrl + and Ctrl −.</p>
            </div>
            <div className="mirid-first-run__zoom" aria-label="Interface size">
              <button type="button" aria-label="Make interface smaller" onClick={() => adjustZoom(zoom - 0.1)} disabled={zoom <= INTERFACE_ZOOM_MIN}>A−</button>
              <button type="button" onClick={() => adjustZoom(INTERFACE_ZOOM_DEFAULT)}>{Math.round(zoom * 100)}%</button>
              <button type="button" aria-label="Make interface larger" onClick={() => adjustZoom(zoom + 0.1)} disabled={zoom >= INTERFACE_ZOOM_MAX}>A+</button>
            </div>
          </div>
        )}

        <div className="mirid-first-run__runtime-note">
          Next, Mirid installs and verifies its local engine. The download is about 3.3 GB and uses about 9 GB after extraction. Chat models are chosen separately.
        </div>
        {error && <div className="mirid-first-run__error" role="alert">Mirid could not begin setup. {error}</div>}
        <footer className="mirid-first-run__footer">
          <span>{purpose ? 'Your choice changes the welcome, not your access.' : 'Choose one path to continue.'}</span>
          <button type="button" className="mirid-first-run__continue" disabled={!purpose || starting} onClick={begin}>
            {starting ? 'Beginning setup…' : 'Install Mirid’s engine'}
          </button>
        </footer>
      </section>
    </main>
  );
}
