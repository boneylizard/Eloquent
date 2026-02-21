import React, { useState, useEffect, useRef, useCallback, memo } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';

// Local state + debounced sync so typing doesn't re-render parent Chat (same pattern as ChatInputForm).
const DEBOUNCE_MS = 400;
const MAX_TOKENS = 150;
const countTokens = (text) => Math.ceil((text || '').length / 4);

const AuthorsNotePanel = memo(function AuthorsNotePanel({
  initialValue,
  onSync,
  onClose,
  visible,
}) {
  const [localValue, setLocalValue] = useState(initialValue || '');
  const debounceRef = useRef(null);
  const isMountedRef = useRef(true);
  const prevVisibleRef = useRef(visible);

  // Sync from parent only when panel first opens (so we don't overwrite while typing)
  useEffect(() => {
    if (visible && !prevVisibleRef.current) {
      setLocalValue(initialValue || '');
    }
    prevVisibleRef.current = visible;
  }, [visible, initialValue]);

  // Debounced sync to parent + localStorage
  const flushSync = useCallback(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    onSync(localValue);
  }, [localValue, onSync]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const handleChange = (e) => {
    const value = e.target.value;
    const tokens = countTokens(value);
    if (tokens > MAX_TOKENS) return;
    setLocalValue(value);

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      debounceRef.current = null;
      if (isMountedRef.current) onSync(value);
    }, DEBOUNCE_MS);
  };

  const handleClose = () => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    onSync(localValue);
    onClose();
  };

  if (!visible) return null;

  const tokenCount = countTokens(localValue);
  return (
    <div className="border-t border-border bg-orange-50 dark:bg-orange-950/20">
      <div className="max-w-4xl mx-auto px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <span className="font-bold text-sm text-orange-600">Author&apos;s Note</span>
          <span className="text-xs text-muted-foreground">{tokenCount}/{MAX_TOKENS} tokens</span>
        </div>
        <Textarea
          value={localValue}
          onChange={handleChange}
          placeholder="Author's note..."
          className="w-full resize-none bg-background text-sm"
          rows={2}
        />
        <Button size="sm" variant="ghost" onClick={handleClose} className="w-full mt-1 text-xs">Close Note</Button>
      </div>
    </div>
  );
});

export default AuthorsNotePanel;
