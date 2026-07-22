/**
 * XML Frame Parser — parses XML frame strings from somatic payloads
 * into structured scene descriptors for the SceneFrameRenderer.
 *
 * XML frames describe structural changes to the scene:
 *   <Scene><Position>pinning against wall</Position></Scene>
 *   <Scene><Location>kitchen</Location><Pose>standing</Pose></Scene>
 *
 * Uses the browser's built-in DOMParser (no external deps).
 */

const PARSER = new DOMParser();

/**
 * Parse an XML frame string into a scene descriptor object.
 * Returns null if parsing fails or the XML is empty.
 *
 * @param {string} xmlStr - The XML frame string
 * @returns {Object|null} - { tag, children: [{ tag, text }] }
 */
export function parseXmlFrame(xmlStr) {
  if (!xmlStr || typeof xmlStr !== 'string' || !xmlStr.trim()) return null;

  try {
    const doc = PARSER.parseFromString(xmlStr.trim(), 'text/xml');
    const parserError = doc.querySelector('parsererror');
    if (parserError) return null;

    const root = doc.documentElement;
    if (!root) return null;

    return _elementToDescriptor(root);
  } catch {
    return null;
  }
}

function _elementToDescriptor(el) {
  const children = [];
  for (const child of el.children) {
    const hasSubChildren = child.children.length > 0;
    if (hasSubChildren) {
      children.push(_elementToDescriptor(child));
    } else {
      children.push({
        tag: child.tagName,
        text: (child.textContent || '').trim(),
      });
    }
  }

  return {
    tag: el.tagName,
    text: children.length === 0 ? (el.textContent || '').trim() : '',
    children,
  };
}

/**
 * Flatten a scene descriptor into a list of { tag, text } pairs
 * for simple rendering.
 */
export function flattenScene(descriptor) {
  if (!descriptor) return [];
  const result = [];
  if (descriptor.text) {
    result.push({ tag: descriptor.tag, text: descriptor.text });
  }
  for (const child of descriptor.children || []) {
    if (child.children && child.children.length > 0) {
      result.push(...flattenScene(child));
    } else if (child.text) {
      result.push({ tag: child.tag, text: child.text });
    }
  }
  return result;
}
