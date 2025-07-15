
import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Copy, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const CodeBlock = React.memo(({ node, inline, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '');
  const codeText = String(children).replace(/\n$/, '');
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(codeText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [codeText]);

  return !inline && match ? (
    <div className="relative group my-4 rounded-md bg-[#282c34] text-sm">
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-600">
        <span className="text-gray-400 text-xs font-sans">{match[1]}</span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 text-gray-400 hover:text-white"
          onClick={handleCopy}
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
        </Button>
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