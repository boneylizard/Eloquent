import * as React from "react";
import { cn } from "../../lib/utils";

const Checkbox = React.forwardRef(({ className, checked, onCheckedChange, ...props }, ref) => {
  const state = checked ? "checked" : "unchecked";
  return (
    <div className="relative inline-flex">
      <input
        type="checkbox"
        ref={ref}
        className={cn(
          "peer h-4 w-4 shrink-0 rounded-sm border border-primary ring-offset-background",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          "disabled:cursor-not-allowed disabled:opacity-50",
          "appearance-none bg-background",
          "checked:bg-primary checked:text-primary-foreground",
          className
        )}
        checked={checked}
        data-state={state}
        onChange={(e) => onCheckedChange?.(e.target.checked)}
        {...props}
      />
      {checked && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-primary-foreground">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-3 w-3"
          >
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </div>
      )}
    </div>
  );
});

Checkbox.displayName = "Checkbox";

export { Checkbox };
