import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  BookOpenText,
  Cable,
  LayoutGrid,
} from 'lucide-react';
import {
  FIRST_RUN_PURPOSES,
  readFirstRunIntent,
  writeFirstRunIntent,
} from '../utils/firstRunIntent';
import './FirstRunPurpose.css';

const ICONS = {
  roleplay: BookOpenText,
  sillytavern: Cable,
  classic: LayoutGrid,
};

export default function FirstRunPurpose({ onBegin }) {
  const [purpose, setPurpose] = useState(() => readFirstRunIntent()?.purpose || '');
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');

  const begin = async () => {
    if (!purpose || starting) return;
    setStarting(true);
    setError('');
    try {
      const intent = writeFirstRunIntent({ purpose });
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
            const Icon = ICONS[item.id] || LayoutGrid;
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

        {purpose === 'sillytavern' && (
          <div className="mirid-first-run__roleplay-note">
            <Cable aria-hidden="true" />
            <p><strong>Close SillyTavern before continuing.</strong> Both programs use port 8000 by default. Mirid's setup guide will move SillyTavern to port 8001 before they run together.</p>
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
