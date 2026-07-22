import React, { useMemo } from 'react';
import { parseXmlFrame, flattenScene } from '../../utils/xmlFrameParser';

export default function SceneFrameRenderer({ xmlFrame }) {
  const sceneItems = useMemo(() => {
    if (!xmlFrame) return [];
    const descriptor = parseXmlFrame(xmlFrame);
    if (!descriptor) return [];
    return flattenScene(descriptor);
  }, [xmlFrame]);

  if (sceneItems.length === 0) return null;

  return (
    <div className="sanctuary-scene-frame">
      {sceneItems.map((item, i) => (
        <div key={i}>
          <span className="sanctuary-scene-frame-tag">{item.tag}: </span>
          <span>{item.text}</span>
        </div>
      ))}
    </div>
  );
}
