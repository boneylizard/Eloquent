// Wraps quoted dialogue in emphasis with a class so it can be styled distinctly.
const remarkDialogueQuotes = () => {
  const skipTypes = new Set(['code', 'inlineCode', 'link', 'linkReference', 'definition', 'image']);

  const splitQuotedText = (text) => {
    if (!text || (!text.includes('"') && !text.includes('“'))) {
      return [{ type: 'text', value: text }];
    }

    const regex = /"[^"\n]+?"|“[^”\n]+?”/g;
    const nodes = [];
    let lastIndex = 0;

    for (const match of text.matchAll(regex)) {
      const start = match.index ?? 0;
      const end = start + match[0].length;
      if (start > lastIndex) {
        nodes.push({ type: 'text', value: text.slice(lastIndex, start) });
      }
      nodes.push({
        type: 'emphasis',
        data: {
          hProperties: {
            className: ['dialogue-quote']
          }
        },
        children: [{ type: 'text', value: match[0] }]
      });
      lastIndex = end;
    }

    if (lastIndex < text.length) {
      nodes.push({ type: 'text', value: text.slice(lastIndex) });
    }

    return nodes.length ? nodes : [{ type: 'text', value: text }];
  };

  const transformNode = (node) => {
    if (!node || !Array.isArray(node.children)) return;
    if (skipTypes.has(node.type)) return;

    const nextChildren = [];
    node.children.forEach((child) => {
      if (child?.type === 'text' && typeof child.value === 'string') {
        nextChildren.push(...splitQuotedText(child.value));
        return;
      }

      if (child?.children) {
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

export default remarkDialogueQuotes;
