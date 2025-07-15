// This code defines a Tabs component with subcomponents for creating a tabbed interface.
// It uses React's Context API to manage the active tab state and provides a simple API for usage.
import React, { createContext, useContext, useState } from 'react';
import { cn } from '../../lib/utils'; // Assuming utils is in lib relative to src

// 1. Create a Context for Tabs state
const TabsContext = createContext({
  activeTab: '',
  setActiveTab: () => {},
});

// Custom hook to use the Tabs context
const useTabs = () => useContext(TabsContext);

// 2. Main Tabs component (Provider)
const Tabs = React.forwardRef(({ className, defaultValue, value, onValueChange, ...props }, ref) => {
  // Internal state if not controlled, otherwise use provided value
  const [internalActiveTab, setInternalActiveTab] = useState(defaultValue || '');

  // Determine active tab: controlled (value prop) or internal state
  const activeTab = value !== undefined ? value : internalActiveTab;

  // Handler to change tab state
  const setActiveTab = (newValue) => {
    if (value === undefined) { // Only set internal state if not controlled
      setInternalActiveTab(newValue);
    }
    if (onValueChange) { // Call external handler if provided
      onValueChange(newValue);
    }
  };

  return (
    // Provide state and handler to children via Context
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div ref={ref} className={cn("tabs-container", className)} {...props} />
    </TabsContext.Provider>
  );
});
Tabs.displayName = "Tabs";

// 3. TabsList component (Layout)
const TabsList = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground",
      className
    )}
    role="tablist" // Accessibility: role for the list
    {...props}
  />
));
TabsList.displayName = "TabsList";

// 4. TabsTrigger component (Button)
const TabsTrigger = React.forwardRef(({ className, value, disabled, children, ...props }, ref) => {
  const { activeTab, setActiveTab } = useTabs(); // Get state/handler from context
  const isActive = activeTab === value;

  const handleClick = () => {
    if (!disabled) {
      setActiveTab(value); // Call the context handler on click
    }
  };

  return (
    <button
      ref={ref}
      type="button" // Ensure it's a button type
      role="tab" // Accessibility: role for the trigger
      aria-selected={isActive} // Accessibility: indicate selected state
      data-state={isActive ? 'active' : 'inactive'} // For styling
      disabled={disabled}
      onClick={handleClick} // Use the simplified handler
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        // Apply active styles based on data-state
        "data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
});
TabsTrigger.displayName = "TabsTrigger";

// 5. TabsContent component (Content Panel)
const TabsContent = React.forwardRef(({ className, value, children, ...props }, ref) => {
  const { activeTab } = useTabs(); // Get active tab value from context
  const isActive = activeTab === value;

  // Render content only if this tab is active
  if (!isActive) {
    return null;
  }

  return (
    <div
      ref={ref}
      role="tabpanel" // Accessibility: role for the content panel
      data-state="active" // For styling active content
      className={cn(
        "mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
});
TabsContent.displayName = "TabsContent";

export { Tabs, TabsList, TabsTrigger, TabsContent };