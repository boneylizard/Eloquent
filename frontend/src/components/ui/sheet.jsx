import * as React from "react"
import * as SheetPrimitive from "@radix-ui/react-dialog";

import { cn } from "@/lib/utils"
import { Cross2Icon } from "@radix-ui/react-icons"

const Sheet = React.forwardRef(({ className, children, open, onOpenChange, ...props }, ref) => (
    <SheetPrimitive.Root
        ref={ref}
        className={cn(className)}
        open={open}
        onOpenChange={onOpenChange}
        {...props}
        >
        {children}
        </SheetPrimitive.Root>
    ));
Sheet.displayName = "Sheet";

const SheetTrigger = SheetPrimitive.Trigger;

const SheetClose = React.forwardRef(({ className, ...props }, ref) => (
    <SheetPrimitive.Close
    ref={ref}
    className={cn(
        "absolute top-4 right-4 rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none",
        className
    )}
    {...props}
    >
    <Cross2Icon className="h-4 w-4" />
    <span className="sr-only">Close</span>
    </SheetPrimitive.Close>
));
SheetClose.displayName = "SheetClose";

const SheetContent = React.forwardRef(({ className, ...props }, ref) => (
        <SheetPrimitive.Content ref={ref} className={cn("fixed z-50 gap-4 bg-background p-6 shadow-lg border", className)} {...props} />
));

SheetContent.displayName = "SheetContent";

const SheetHeader = React.forwardRef(({ className, ...props }, ref) => (
    <SheetPrimitive.Header ref={ref} className={cn("flex flex-col space-y-2 text-center sm:text-left", className)} {...props} />
));
SheetHeader.displayName = "SheetHeader";

const SheetFooter = React.forwardRef(({ className, ...props }, ref) => (
    <SheetPrimitive.Footer ref={ref} className={cn(
        "flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2",
        className
    )} {...props} />
));
SheetFooter.displayName = "SheetFooter";

const SheetTitle = React.forwardRef(({ className, ...props }, ref) => (
    <SheetPrimitive.Title ref={ref} className={cn(
        "text-lg font-semibold text-foreground",
        className
    )} {...props} />
));
SheetTitle.displayName = "SheetTitle";

const SheetDescription = React.forwardRef(({ className, ...props }, ref) => (
    <SheetPrimitive.Description ref={ref} className={cn(
        "text-sm text-muted-foreground",
        className
    )} {...props} />
));
SheetDescription.displayName = "SheetDescription";


export { Sheet, SheetTrigger, SheetContent, SheetClose, SheetHeader, SheetFooter, SheetTitle, SheetDescription };