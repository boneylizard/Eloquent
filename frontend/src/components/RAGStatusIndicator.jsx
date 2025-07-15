import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { BookOpen, AlertCircle, Loader2 } from 'lucide-react';
import { 
  Tooltip, 
  TooltipContent, 
  TooltipProvider, 
  TooltipTrigger 
} from './ui/tooltip';

const RAGStatusIndicator = () => {
  const { PRIMARY_API_URL, settings } = useApp();
  const [ragStatus, setRagStatus] = useState({
    available: false,
    loading: true,
    message: "Checking RAG status..."
  });

  useEffect(() => {
    const checkRagStatus = async () => {
      try {
        const res = await fetch(`${PRIMARY_API_URL}/rag/status`);
        if (res.ok) {
          const data = await res.json();
          setRagStatus({
            available: data.available,
            loading: false,
            message: data.message
          });
        } else {
          setRagStatus({
            available: false,
            loading: false,
            message: "Failed to check RAG status"
          });
        }
      } catch (error) {
        console.error("Error checking RAG status:", error);
        setRagStatus({
          available: false,
          loading: false,
          message: "Error checking RAG status"
        });
      }
    };

    checkRagStatus();
  }, [PRIMARY_API_URL]);

  const getIndicatorColor = () => {
    if (ragStatus.loading) return "text-yellow-500";
    if (!ragStatus.available) return "text-red-500";
    if (settings.use_rag) return "text-green-500";
    return "text-gray-400";
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="cursor-help">
            {ragStatus.loading ? (
              <Loader2 className={`h-5 w-5 animate-spin ${getIndicatorColor()}`} />
            ) : (
              <BookOpen className={`h-5 w-5 ${getIndicatorColor()}`} />
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          <div className="max-w-xs">
            <div className="font-medium">Document Integration</div>
            <p className="text-sm">
              {ragStatus.loading
                ? "Checking if document augmentation is available..."
                : ragStatus.available
                  ? settings.use_rag
                    ? `Active: ${settings.selectedDocuments?.length || 0} documents selected`
                    : "Available but not enabled"
                  : "Not available - missing server dependencies"
              }
            </p>
            {ragStatus.message && !ragStatus.loading && (
              <p className="text-xs text-muted-foreground mt-1">{ragStatus.message}</p>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default RAGStatusIndicator;