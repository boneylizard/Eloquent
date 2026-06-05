import React from 'react';
import { useApp } from '../contexts/AppContext';
import DocumentSelector from './DocumentSelector';
import { Checkbox } from './ui/checkbox';
import { Label } from './ui/label';
import { Button } from './ui/button';

/**
 * Shared RAG controls for Chatlog condenser (quick / agent / autonomous).
 * Upload transcript via Settings → Documents, then select docs here or sync from chat.
 */
export default function ChatlogCondenserRagOptions({
  useRag,
  onUseRagChange,
  selectedDocs,
  onSelectedDocsChange,
}) {
  const { settings = {}, documents } = useApp();
  const docCount = documents?.file_list?.length ?? 0;
  const activeFromChat = settings.selectedDocuments || [];
  const hasActiveChatDocs = settings.use_rag && activeFromChat.length > 0;

  const useActiveFromChat = () => {
    if (!activeFromChat.length) return;
    onSelectedDocsChange([...activeFromChat]);
    onUseRagChange(true);
  };

  const selectedLabels = (documents?.file_list || [])
    .filter((d) => selectedDocs.includes(d.id))
    .map((d) => d.filename);

  return (
    <div className="rounded-md border p-3 space-y-3 bg-muted/30">
      <div className="flex items-start gap-2">
        <Checkbox
          id="condenser_use_rag"
          checked={useRag}
          onCheckedChange={(c) => onUseRagChange(c === true)}
          disabled={docCount === 0 && !selectedDocs.length}
        />
        <div className="space-y-1">
          <Label htmlFor="condenser_use_rag" className="cursor-pointer font-medium">
            Supplement with document context (RAG)
          </Label>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Cross-reference uploaded transcript chunks per step — does not replace sequential
            turn order. Best with full log context off on very long logs.
          </p>
        </div>
      </div>
      {useRag && (
        <div className="space-y-2 pl-6">
          <DocumentSelector
            selectedDocs={selectedDocs}
            onChange={onSelectedDocsChange}
            maxSelections={8}
          />
          {hasActiveChatDocs && (
            <Button type="button" variant="outline" size="sm" onClick={useActiveFromChat}>
              Use active chat document selection ({activeFromChat.length})
            </Button>
          )}
          {selectedDocs.length > 0 && (
            <p className="text-xs text-muted-foreground">
              Selected: {selectedLabels.join(', ') || `${selectedDocs.length} document(s)`}
            </p>
          )}
          {selectedDocs.length === 0 && (
            <p className="text-xs text-amber-600 dark:text-amber-500">
              Select at least one document (upload .txt/.md of the transcript in Documents first).
            </p>
          )}
        </div>
      )}
    </div>
  );
}
