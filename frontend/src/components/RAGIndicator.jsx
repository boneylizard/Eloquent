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
            <Badge variant="outline" className="bg-yellow-50 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 dark:border-yellow-800/30">
              <AlertCircle className="h-3 w-3 mr-1" />
              Document Context: No Files Selected
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            <p>Document context is enabled but no documents are selected in Settings</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }
  
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant="outline" className="bg-green-50 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-400 dark:border-green-800/30">
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