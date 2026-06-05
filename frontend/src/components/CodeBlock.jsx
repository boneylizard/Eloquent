
import React, { useState, useCallback, useMemo, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Copy, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useApp } from '@/contexts/AppContext';

const INSTRUCTIONS_SEPARATOR = '\n\n---\n\n';

function appendToModelInstructions(existing, chunk) {
  const trimmed = String(chunk ?? '').trim();
  if (!trimmed) return existing ?? '';
  const base = String(existing ?? '').trimEnd();
  if (!base) return trimmed;
  return `${base}${INSTRUCTIONS_SEPARATOR}${trimmed}`;
}

const CodeBlock = React.memo(({ node, inline, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '');
  const codeText = String(children).replace(/\n$/, '');
  const [copied, setCopied] = useState(false);
  const [appendStatus, setAppendStatus] = useState(null);
  const appendStatusTimerRef = useRef(null);

  const {
    activeCharacter,
    primaryCharacter,
    saveCharacter,
    setActiveCharacter,
    setPrimaryCharacter,
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

  return !inline && match ? (
    <div className="relative group my-4 rounded-md bg-[#282c34] text-sm">
      <div className="flex items-center justify-between gap-2 px-4 py-2 border-b border-gray-600">
        <span className="text-gray-400 text-xs font-sans shrink-0">{match[1]}</span>
        <div className="flex items-center gap-1 min-w-0">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className={cn(
              'h-6 px-2 text-xs text-gray-400 hover:text-white shrink-0',
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
            className="h-6 w-6 text-gray-400 hover:text-white shrink-0"
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
  ) : (
    <code className={cn("font-mono text-sm bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded", className)} {...props}>
      {children}
    </code>
  );
});

export default CodeBlock;
