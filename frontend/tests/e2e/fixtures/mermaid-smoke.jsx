import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { MermaidBlock } from '../../../src/components/CodeBlock';

function MermaidSmoke() {
  const [isGenerating, setIsGenerating] = useState(true);

  useEffect(() => {
    const finishTimer = setTimeout(() => setIsGenerating(false), 300);
    return () => clearTimeout(finishTimer);
  }, []);

  return (
    <MermaidBlock
      code={'flowchart LR\n  start[Start] --> finish[Done]'}
      isGenerating={isGenerating}
    />
  );
}

createRoot(document.getElementById('root')).render(<MermaidSmoke />);
