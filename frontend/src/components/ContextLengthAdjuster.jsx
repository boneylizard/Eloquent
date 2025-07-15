import React, { useState, useEffect } from 'react';

const ContextLengthAdjuster = ({ defaultValue = 4096, min = 2048, max = 32768, onChange }) => {
  const [contextLength, setContextLength] = useState(defaultValue);
  
  // Predefined context lengths to choose from
  const contextPresets = [
    { label: '4K', value: 4096 },
    { label: '8K', value: 8192 },
    { label: '16K', value: 16384 },
    { label: '32K', value: 32768 }
  ];
  
  useEffect(() => {
    // Call onChange when component mounts with default value
    if (onChange) {
      onChange(defaultValue);
    }
  }, []);
  
  const handleSliderChange = (e) => {
    const newValue = parseInt(e.target.value, 10);
    setContextLength(newValue);
    if (onChange) {
      onChange(newValue);
    }
  };
  
  const handlePresetClick = (value) => {
    setContextLength(value);
    if (onChange) {
      onChange(value);
    }
  };
  
  // Format the number for display (e.g., 4096 → "4K")
  const formatContextLength = (length) => {
    if (length >= 1024) {
      return `${(length / 1024).toFixed(0)}K`;
    }
    return length.toString();
  };
  
  return (
    <div className="context-length-adjuster">
      <div className="context-length-label">
        Context Length: <span className="context-length-value">{formatContextLength(contextLength)}</span>
      </div>
      
      <div className="context-length-slider-container">
        <input
          type="range"
          min={min}
          max={max}
          step={1024}
          value={contextLength}
          onChange={handleSliderChange}
          className="context-length-slider"
        />
      </div>
      
      <div className="context-presets">
        {contextPresets.map((preset) => (
          <button
            key={preset.value}
            className={`preset-button ${contextLength === preset.value ? 'active' : ''}`}
            onClick={() => handlePresetClick(preset.value)}
          >
            {preset.label}
          </button>
        ))}
      </div>
      
      <style jsx>{`
        .context-length-adjuster {
          margin: 10px 0;
          padding: 10px;
          border-radius: 8px;
          background-color: #f5f5f5;
        }
        
        .context-length-label {
          font-size: 14px;
          margin-bottom: 6px;
        }
        
        .context-length-value {
          font-weight: bold;
        }
        
        .context-length-slider-container {
          padding: 0 5px;
        }
        
        .context-length-slider {
          width: 100%;
          height: 4px;
        }
        
        .context-presets {
          display: flex;
          justify-content: space-between;
          margin-top: 10px;
        }
        
        .preset-button {
          padding: 4px 8px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
          cursor: pointer;
          font-size: 12px;
        }
        
        .preset-button.active {
          background-color: #007bff;
          color: white;
          border-color: #007bff;
        }
      `}</style>
    </div>
  );
};

export default ContextLengthAdjuster;