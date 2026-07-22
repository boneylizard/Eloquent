import * as React from "react";
import { cn } from "../../lib/utils";

const Slider = React.forwardRef(
  ({ className, min = 0, max = 100, step = 1, defaultValue, value, onValueChange, ...props }, ref) => {
    const [localValue, setLocalValue] = React.useState(defaultValue ?? [min]);

    React.useEffect(() => {
      if (value !== undefined) {
        setLocalValue(value);
      }
    }, [value]);

    const handleChange = (e) => {
      const newValue = [Number(e.target.value)];
      setLocalValue(newValue);
      onValueChange?.(newValue);
    };

    const clamped = typeof localValue?.[0] === "number" ? localValue[0] : min;
    const span = max - min || 1;
    const percent = Math.max(0, Math.min(100, ((clamped - min) / span) * 100));

    return (
      <div className={cn("relative flex w-full touch-none select-none items-center", className)}>
        <div
          className="absolute h-2 w-full rounded-full bg-muted pointer-events-none"
          aria-hidden="true"
        />
        <div
          className="absolute h-2 rounded-full bg-primary pointer-events-none"
          style={{ width: `${percent}%` }}
          aria-hidden="true"
        />
        <input
          ref={ref}
          type="range"
          min={min}
          max={max}
          step={step}
          value={clamped}
          onChange={handleChange}
          className={cn(
            "relative w-full h-2 appearance-none bg-transparent outline-none cursor-pointer",
            "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
            "disabled:opacity-50 disabled:cursor-not-allowed",
            className
          )}
          {...props}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 h-4 w-4 rounded-full bg-primary border-2 border-background pointer-events-none shadow-sm transition-transform"
          style={{ left: `calc(${percent}% - 8px)` }}
          aria-hidden="true"
        />
      </div>
    );
  }
);

Slider.displayName = "Slider";

export { Slider };
