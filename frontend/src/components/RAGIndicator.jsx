// Create a new file called RAGIndicator.jsx
import React from 'react';
import { useApp } from '../contexts/AppContext';
import { FileText, AlertCircle } from 'lucide-react';
import { Badge } from './ui/badge';
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from './ui/tooltip';

const RAGIndicator = () => {
  const { settings } = useApp();
  
  if (!settings.use_rag) {
    return null;
  }
  
  const docCount = (settings.selectedDocuments || []).length;
  
  if (docCount === 0) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge variant="outline" className="border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400">
              <AlertCircle className="h-3 w-3 mr-1" />
              Document Context: No Files Selected
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            <p>Document context is on, but no files are checked for context. Open Documents to select some.</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant="outline" className="border-primary/40 bg-primary/10 text-primary">
            <FileText className="h-3 w-3 mr-1" />
            Document Context: {docCount} {docCount === 1 ? 'file' : 'files'}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <p>LLM responses will be enhanced with document context</p>
          {docCount > 0 && (
            <ul className="mt-1 text-xs">
              {settings.selectedDocuments.map((docId, index) => (
                <li key={docId}>• Document {index + 1}</li>
              ))}
            </ul>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default RAGIndicator;