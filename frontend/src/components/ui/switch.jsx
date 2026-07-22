import * as React from "react";
import { cn } from "../../lib/utils";

const Switch = React.forwardRef(({ className, checked, defaultChecked, onCheckedChange, ...props }, ref) => {
  const [isChecked, setIsChecked] = React.useState(defaultChecked ?? false);

  React.useEffect(() => {
    if (checked !== undefined) {
      setIsChecked(checked);
    }
  }, [checked]);

  const handleChange = (e) => {
    const next = e.target.checked;
    setIsChecked(next);
    onCheckedChange?.(next);
  };

  return (
    <label className={cn("relative inline-flex items-center cursor-pointer", className)}>
      <input
        type="checkbox"
        ref={ref}
        checked={isChecked}
        onChange={handleChange}
        className="sr-only peer"
        {...props}
      />
      <div
        className={cn(
          "relative w-11 h-6 bg-muted rounded-full peer transition-colors",
          "peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-ring peer-focus:ring-offset-2 peer-focus:ring-offset-background",
          "peer-checked:bg-primary",
          "after:content-[''] after:absolute after:top-[2px] after:left-[2px]",
          "after:bg-background after:border-border after:border after:rounded-full after:h-5 after:w-5",
          "after:shadow-sm after:transition-all peer-checked:after:translate-x-full"
        )}
      />
    </label>
  );
});

Switch.displayName = "Switch";

export { Switch };
