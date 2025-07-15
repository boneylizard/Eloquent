import React, { useState } from 'react';
import { cn } from '../../lib/utils';

// No TooltipProvider needed for now - REMOVE, it was redundant

const Tooltip = ({ children, open, defaultOpen, onOpenChange, ...props }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen || false);

  const handleOpenChange = (open) => {
    setIsOpen(open);
    onOpenChange?.(open); // This is fine; it's used internally
  };

  return (
    // TooltipPrimitive was renamed to TooltipPrimitiveComponent
    <TooltipPrimitiveComponent
      open={open !== undefined ? open : isOpen}
      onOpenChange={handleOpenChange} // Pass it down, BUT...
      {...props}
    >
      {children}
    </TooltipPrimitiveComponent>
  );
};

// ... but remove it from here:
const TooltipPrimitiveComponent = React.forwardRef(({ children, className, open, ...props }, ref) => { // Removed onOpenChange
  return (
    <div ref={ref} className={cn("tooltip-primitive", className)} {...props}>
      {children}
    </div>
  );
});
TooltipPrimitiveComponent.displayName = "TooltipPrimitive"; // Use consistent name


const TooltipTrigger = React.forwardRef(({ className, asChild = false, children, ...props }, ref) => {
  if (asChild) {
    return React.cloneElement(children, {
      ref: ref,
      className: cn("", className, children.props.className), // Combine classNames
      ...props,
    });
  }

  return (
    <button ref={ref} className={cn("", className)} {...props}>
      {children}
    </button>
  );
});
TooltipTrigger.displayName = "TooltipTrigger";

const TooltipContent = React.forwardRef(({ className, sideOffset = 4, open, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md",
        open ? "animate-in fade-in-50 data-[side=bottom]:slide-in-from-top-1 data-[side=top]:slide-in-from-bottom-1 data-[side=left]:slide-in-from-right-1 data-[side=right]:slide-in-from-left-1" : "hidden",
        className
      )}
      {...props}
    />
  );
});
TooltipContent.displayName = "TooltipContent";
// TooltipProvider was removed
export { Tooltip, TooltipTrigger, TooltipContent };