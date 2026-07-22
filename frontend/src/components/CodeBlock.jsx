
import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Copy, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useApp } from '@/contexts/AppContext';

let mermaidPromise;

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
        suppressErrorRendering: true,
      });
      return mermaid;
    });
  }
  return mermaidPromise;
}

const INSTRUCTIONS_SEPARATOR = '\n\n---\n\n';

function appendToModelInstructions(existing, chunk) {
  const trimmed = String(chunk ?? '').trim();
  if (!trimmed) return existing ?? '';
  const base = String(existing ?? '').trimEnd();
  if (!base) return trimmed;
  return `${base}${INSTRUCTIONS_SEPARATOR}${trimmed}`;
}

let mermaidCounter = 0;

export const MermaidBlock = React.memo(({ code, isGenerating }) => {
  const containerRef = useRef(null);
  const [status, setStatus] = useState('waiting');
  const [error, setError] = useState(null);
  const idRef = useRef(`mirid-mermaid-${++mermaidCounter}`);
  const trimmedCode = String(code || '').trim();

  useEffect(() => {
    if (!containerRef.current || isGenerating) {
      if (!containerRef.current?.hasChildNodes()) setStatus('waiting');
      return undefined;
    }

    let cancelled = false;
    setStatus('loading');
    setError(null);

    const renderTimer = setTimeout(async () => {
      try {
        const mermaid = await loadMermaid();
        if (cancelled) return;
        await mermaid.parse(trimmedCode);
        const { svg, bindFunctions } = await mermaid.render(idRef.current, trimmedCode);
        if (cancelled || !containerRef.current) return;
        containerRef.current.innerHTML = svg;
        bindFunctions?.(containerRef.current);
        const svgElement = containerRef.current.querySelector('svg');
        if (svgElement) {
          svgElement.style.maxWidth = 'none';
          svgElement.style.height = 'auto';
        }
        setStatus('ready');
      } catch (renderError) {
        if (cancelled) return;
        setError(renderError?.message || 'Mirid could not render this diagram.');
        setStatus('error');
      }
    }, 350);

    return () => {
      cancelled = true;
      clearTimeout(renderTimer);
    };
  }, [trimmedCode, isGenerating]);

  return (
    <div className="my-4 rounded-md p-4 overflow-x-auto" style={{ backgroundColor: 'var(--chat-code-bg)' }}>
      {status === 'waiting' && (
        <div className="text-xs text-gray-400 font-sans">Finishing the diagram before rendering it.</div>
      )}
      {status === 'loading' && (
        <div className="text-xs text-gray-400 font-sans">Rendering diagram…</div>
      )}
      {status === 'error' && (
        <div className="mb-3 text-xs text-red-400 font-sans">{error}</div>
      )}
      <div ref={containerRef} className="flex justify-center" style={{ color: 'initial' }} />
      {status === 'error' && (
        <pre className="mt-3 text-sm text-gray-300 whitespace-pre-wrap">{code}</pre>
      )}
    </div>
  );
});
MermaidBlock.displayName = 'MermaidBlock';

const CodeBlock = React.memo(({ node, inline, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '');
  const codeText = (typeof children === 'string'
    ? children
    : Array.isArray(children)
      ? children.map(c => typeof c === 'string' ? c : '').join('')
      : ''
  ).replace(/\n$/, '');
  const [copied, setCopied] = useState(false);
  const [appendStatus, setAppendStatus] = useState(null);
  const appendStatusTimerRef = useRef(null);

  const {
    activeCharacter,
    primaryCharacter,
    saveCharacter,
    setActiveCharacter,
    setPrimaryCharacter,
    isGenerating,
  } = useApp();

  const targetCharacter = useMemo(
    () => activeCharacter || primaryCharacter,
    [activeCharacter, primaryCharacter]
  );

  const canAppend = Boolean(targetCharacter?.id);
  const appendDisabledReason = !canAppend
    ? 'Select an active character to append'
    : !codeText.trim()
      ? 'Code block is empty'
      : null;

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(codeText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [codeText]);

  const flashAppendStatus = useCallback((status) => {
    setAppendStatus(status);
    if (appendStatusTimerRef.current) clearTimeout(appendStatusTimerRef.current);
    appendStatusTimerRef.current = setTimeout(() => setAppendStatus(null), 2200);
  }, []);

  const handleAppendToCharacter = useCallback(() => {
    if (!codeText.trim()) {
      flashAppendStatus('empty');
      return;
    }
    if (!targetCharacter?.id) {
      flashAppendStatus('no-char');
      return;
    }

    const nextInstructions = appendToModelInstructions(
      targetCharacter.model_instructions,
      codeText
    );
    const updated = saveCharacter({
      ...targetCharacter,
      model_instructions: nextInstructions,
    });

    if (activeCharacter?.id === updated.id) {
      setActiveCharacter(updated);
    }
    if (primaryCharacter?.id === updated.id) {
      setPrimaryCharacter(updated);
    }

    flashAppendStatus('success');
  }, [
    codeText,
    targetCharacter,
    saveCharacter,
    activeCharacter?.id,
    primaryCharacter?.id,
    setActiveCharacter,
    setPrimaryCharacter,
    flashAppendStatus,
  ]);

  const appendLabel = appendStatus === 'success'
    ? 'Appended'
    : appendStatus === 'empty'
      ? 'Empty'
      : appendStatus === 'no-char'
        ? 'No character'
        : 'Append to character';

  if (!inline && match) {
    if (match[1].toLowerCase() === 'mermaid') {
      return (
        <div className="relative group">
          <div className="absolute top-2 right-2 z-10">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-gray-400 hover:text-white"
              onClick={handleCopy}
              title="Copy diagram source"
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </Button>
          </div>
          <MermaidBlock code={codeText} isGenerating={isGenerating} />
        </div>
      );
    }

    return (
      <div className="relative group my-4 rounded-md text-sm" style={{ backgroundColor: 'var(--chat-code-bg)' }}>
        <div className="flex items-center justify-between gap-2 px-4 py-2 border-b" style={{ borderColor: 'var(--chat-code-border)' }}>
          <span className="inline-flex items-center gap-1.5 text-gray-400 text-[10px] font-sans shrink-0 uppercase tracking-wider bg-white/5 rounded-full px-2 py-0.5">{match[1]}</span>
          <div className="flex items-center gap-1 min-w-0">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className={cn(
                'h-6 px-2 text-xs text-gray-400 hover:text-white shrink-0 transition-colors',
                appendStatus === 'success' && 'text-green-400 hover:text-green-300'
              )}
              onClick={handleAppendToCharacter}
              disabled={!canAppend}
              title={appendDisabledReason || (appendStatus === 'success' ? `Appended to ${targetCharacter.name}` : `Append to ${targetCharacter?.name || 'character'} model instructions`)}
            >
              {appendLabel}
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-gray-400 hover:text-white shrink-0 active:scale-90 transition-all"
              onClick={handleCopy}
              title="Copy code"
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </Button>
          </div>
        </div>
        <SyntaxHighlighter style={oneDark} language={match[1]} PreTag="div" {...props}>
          {codeText}
        </SyntaxHighlighter>
      </div>
    );
  }

  return (
    <code className={cn("font-mono text-sm bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded", className)} {...props}>
      {children}
    </code>
  );
});

export default CodeBlock;
