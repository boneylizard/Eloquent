import React, { useState, useEffect, useRef } from 'react';
import { cn } from '../../lib/utils';

const Slider = React.forwardRef(
  ({ className, min = 0, max = 100, step = 1, defaultValue, value, onValueChange, ...props }, ref) => {
    const [localValue, setLocalValue] = useState(defaultValue || [min]);
    const rangeRef = useRef(null);

    // Update local value when controlled value changes
    useEffect(() => {
      if (value !== undefined) {
        setLocalValue(value);
      }
    }, [value]);

    const handleChange = (e) => {
      const newValue = [Number(e.target.value)];
      setLocalValue(newValue);

      if (onValueChange) {
        onValueChange(newValue);
      }
    };

    // Calculate the percentage for styling the track
    const percent = ((localValue[0] - min) / (max - min)) * 100;

    return (
      <div className={cn("relative flex w-full touch-none select-none items-center", className)}>
        <input
          ref={ref}
          type="range"
          min={min}
          max={max}
          step={step}
          value={typeof localValue?.[0] === 'number' ? localValue[0] : min}
          onChange={handleChange}
          className={cn(
            "w-full h-2 appearance-none rounded-full bg-muted outline-none",
             // Add Tailwind classes for the "track" (the background)
            "focus:ring-2 focus:ring-offset-2 focus:ring-ring", // Add focus styles
            "disabled:opacity-50 disabled:cursor-not-allowed", // Add disabled styles
            `before:absolute before:inset-0 before:bg-gradient-to-r before:from-primary before:to-primary before:h-full before:rounded-full`,
            `before:w-[${percent}%]`, //Dynamically apply width to the "track" before the thumb
          )}

          style={{
              '--percent': percent + '%', // Use a CSS variable for the gradient
          }}
          {...props}
        />
        {/*  Remove the <style> tag entirely */}
        <div className="absolute left-0 top-1/2 -translate-y-1/2 h-4 w-4 rounded-full bg-primary border-2 border-background cursor-pointer transition-all"
             style={{
                 left: `calc(${percent}% - 8px)`
             }}>
        </div>
      </div>
    );
  }
);

Slider.displayName = "Slider";

export { Slider };