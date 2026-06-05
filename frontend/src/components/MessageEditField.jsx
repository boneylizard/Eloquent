// Isolated edit field so only this component re-renders on keystroke (avoids lag in parent message list)
import React, { memo, useState, useEffect, useCallback, useRef } from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';

const MessageEditField = memo(function MessageEditField({
  initialValue,
  messageId,
  onSave,
  onCancel,
  onSaveAndRegenerate,
  rows = 4,
  saveLabel = 'Save',
  showSaveAndRegenerate = false,
  disabledSaveAndRegenerate = false,
  className = '',
  textareaClassName = '',
}) {
  const [value, setValue] = useState(initialValue);
  const fileInputRef = useRef(null);

  useEffect(() => {
    setValue(initialValue);
  }, [initialValue, messageId]);

  const handleSave = useCallback(() => {
    onSave(messageId, value.trim());
  }, [onSave, messageId, value]);

  const handleSaveAndRegenerate = useCallback(() => {
    onSave(messageId, value.trim());
    setTimeout(() => onSaveAndRegenerate(messageId, value.trim()), 100);
  }, [onSave, onSaveAndRegenerate, messageId, value]);

  const handleOpenFilePicker = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleImportTxt = useCallback(async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      setValue(text);
    } catch (error) {
      console.error('Failed to import .txt file for message edit:', error);
    } finally {
      // Allow re-selecting the same file on the next import.
      event.target.value = '';
    }
  }, []);

  return (
    <div className={`space-y-2 ${className}`}>
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,text/plain"
        className="hidden"
        onChange={handleImportTxt}
      />
      <Textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className={`w-full resize-none bg-background border-input ${textareaClassName}`}
        rows={rows}
        autoFocus
      />
      <div className="flex gap-2 justify-end">
        <Button variant="outline" size="sm" onClick={handleOpenFilePicker}>
          Import .txt
        </Button>
        <Button variant="outline" size="sm" onClick={onCancel}>
          Cancel
        </Button>
        <Button
          variant="default"
          size="sm"
          onClick={handleSave}
          disabled={!value.trim()}
        >
          {saveLabel}
        </Button>
        {showSaveAndRegenerate && (
          <Button
            variant="secondary"
            size="sm"
            onClick={handleSaveAndRegenerate}
            disabled={!value.trim() || disabledSaveAndRegenerate}
          >
            Save & Regenerate
          </Button>
        )}
      </div>
    </div>
  );
});

export default MessageEditField;
