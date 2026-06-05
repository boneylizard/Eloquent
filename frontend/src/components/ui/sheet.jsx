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

const sideVariants = {
  top: "inset-x-0 top-0 border-b data-[state=closed]:slide-out-to-top data-[state=open]:slide-in-from-top",
  bottom: "inset-x-0 bottom-0 border-t data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom",
  left: "inset-y-0 left-0 h-full w-3/4 border-r data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left sm:max-w-sm",
  right: "inset-y-0 right-0 h-full w-3/4 border-l data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right sm:max-w-md",
};

const SheetContent = React.forwardRef(({ side = "right", className, children, ...props }, ref) => (
        <SheetPrimitive.Portal>
            <SheetPrimitive.Overlay className="fixed inset-0 z-50 bg-black/80 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
            <SheetPrimitive.Content ref={ref} className={cn("fixed z-50 gap-4 bg-background p-6 shadow-lg transition ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:duration-300 data-[state=open]:duration-500", sideVariants[side], className)} {...props}>
                {children}
                <SheetPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-secondary">
                    <Cross2Icon className="h-4 w-4" />
                    <span className="sr-only">Close</span>
                </SheetPrimitive.Close>
            </SheetPrimitive.Content>
        </SheetPrimitive.Portal>
));
SheetContent.displayName = "SheetContent";

const SheetHeader = React.forwardRef(({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex flex-col space-y-2 text-center sm:text-left", className)} {...props} />
));
SheetHeader.displayName = "SheetHeader";

const SheetFooter = React.forwardRef(({ className, ...props }, ref) => (
    <div ref={ref} className={cn(
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