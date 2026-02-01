// Converts soft line breaks into hard breaks without extra dependencies.
// This mirrors remark-breaks behavior for our markdown renderer.
const remarkSoftBreaks = () => {
  const transformNode = (node) => {
    if (!node || !Array.isArray(node.children)) return;

    const nextChildren = [];
    node.children.forEach((child) => {
      if (child && child.type === 'text' && typeof child.value === 'string' && child.value.includes('\n')) {
        const parts = child.value.split('\n');
        parts.forEach((part, index) => {
          if (part) {
            nextChildren.push({ type: 'text', value: part });
          }
          if (index < parts.length - 1) {
            nextChildren.push({ type: 'break' });
          }
        });
        return;
      }

      if (child && Array.isArray(child.children)) {
        transformNode(child);
      }
      nextChildren.push(child);
    });

    node.children = nextChildren;
  };

  return (tree) => {
    transformNode(tree);
  };
};

export default remarkSoftBreaks;
