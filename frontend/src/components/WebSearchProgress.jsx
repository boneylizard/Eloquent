import React, { useState, useEffect } from 'react';
import { Search, Globe, CheckCircle2, Loader2, FileText, ExternalLink, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

const WebSearchProgress = ({ progress, sources = [], isComplete = false }) => {
  const [expanded, setExpanded] = useState(true);
  const [animationKey, setAnimationKey] = useState(0);

  useEffect(() => {
    if (progress?.queries) {
      setAnimationKey(prev => prev + 1);
    }
  }, [progress?.queries]);

  if (!progress && !sources.length && !isComplete) return null;

  const round = progress?.round || 1;
  const queries = progress?.queries || [];
  const currentQuery = queries[0]?.query || '';
  const isDocumentSearch = progress?.kind === 'documents';

  return (
    <div className="my-3 rounded-lg border border-primary/30 bg-accent/20 overflow-hidden animate-fade-in-up">
      <button
        type="button"
        className="flex w-full items-center gap-3 p-3 cursor-pointer hover:bg-accent/40 transition-colors"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        aria-controls="web-search-details"
      >
        <div className="relative flex-shrink-0">
          {isComplete ? (
            <CheckCircle2 className="w-5 h-5 text-primary" />
          ) : isDocumentSearch ? (
            <FileText className="w-5 h-5 text-primary animate-pulse" />
          ) : (
            <>
              <Globe className="w-5 h-5 text-primary animate-pulse" />
              <div className="absolute inset-0 w-5 h-5 border-2 border-primary/60 rounded-full animate-ping opacity-75" />
            </>
          )}
        </div>

        <div className="flex-1 min-w-0 text-left">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-foreground">
              {isComplete ? 'Search complete' : isDocumentSearch ? 'Searching documents…' : 'Searching the web…'}
            </span>
            {!isComplete && (
              <span className="text-xs text-primary bg-primary/15 px-2 py-0.5 rounded-full">
                Round {round}
              </span>
            )}
          </div>
          {!isComplete && currentQuery && (
            <p className="text-xs text-muted-foreground truncate mt-0.5">
              {currentQuery}
            </p>
          )}
          {isComplete && sources.length > 0 && (
            <p className="text-xs text-muted-foreground mt-0.5">
              Found {sources.length} source{sources.length !== 1 ? 's' : ''}
            </p>
          )}
        </div>

        <ChevronDown
          className={cn(
            "w-4 h-4 text-muted-foreground transition-transform duration-200 flex-shrink-0",
            expanded && "rotate-180"
          )}
        />
      </button>

      {expanded && (
        <div
          id="web-search-details"
          className="border-t border-border bg-background/40"
        >
          {!isComplete && queries.length > 0 && (
            <div className="p-3 border-b border-border">
              <div className="text-xs font-medium text-foreground mb-2 flex items-center gap-1.5">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                {isDocumentSearch ? 'Document queries' : 'Active searches'}
              </div>
              <div className="space-y-1.5">
                {queries.map((q, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Search className="w-3 h-3 opacity-60" />
                    <span className="truncate">{q.query || q.tool}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {sources.length > 0 && (
            <div className="p-3">
              <div className="text-xs font-medium text-foreground mb-2 flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5" />
                Sources
              </div>
              <div className="space-y-1.5 max-h-40 overflow-y-auto">
                {sources.slice(0, 8).map((source, idx) => (
                  <a
                    key={idx}
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-2 p-2 rounded-md hover:bg-accent/40 transition-colors group"
                  >
                    <div className="flex-shrink-0 w-5 h-5 rounded-full bg-primary/15 flex items-center justify-center text-xs font-medium text-primary">
                      {idx + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-foreground truncate group-hover:text-primary transition-colors">
                        {source.title || source.url}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">
                        {(() => {
                          try {
                            return new URL(source.url).hostname;
                          } catch {
                            return source.url;
                          }
                        })()}
                      </p>
                    </div>
                    <ExternalLink className="w-3.5 h-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                  </a>
                ))}
                {sources.length > 8 && (
                  <p className="text-xs text-muted-foreground text-center pt-1">
                    +{sources.length - 8} more sources
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {!isComplete && (
        <div className="h-1 bg-muted overflow-hidden">
          <div
            key={animationKey}
            className="h-full bg-primary animate-progress"
          />
        </div>
      )}
    </div>
  );
};

export default WebSearchProgress;
