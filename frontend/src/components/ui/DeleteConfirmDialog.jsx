import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import { Button } from './button';
import { Checkbox } from './checkbox';
import { Label } from './label';

const DeleteConfirmDialog = ({ isOpen, onClose, onConfirm, title, position = { x: 0, y: 0 } }) => {
  const [dontAskAgain, setDontAskAgain] = useState(false);
  
  const handleConfirm = () => {
    onConfirm(dontAskAgain);
    onClose();
  };

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };
  
  if (!isOpen) return null;

  // Calculate position - ensure dialog stays within viewport
  const dialogWidth = 320;
  const dialogHeight = 200;
  const padding = 16;
  
  // Position dialog to the right of the click, or left if no space
  let left = position.x + 10;
  let top = position.y - dialogHeight / 2;
  
  // Adjust if dialog would go off-screen (right edge)
  if (left + dialogWidth + padding > window.innerWidth) {
    left = position.x - dialogWidth - 10; // Show to the left instead
  }
  // Adjust if still off-screen (left edge)
  if (left < padding) {
    left = padding;
  }
  // Adjust vertical position
  if (top + dialogHeight + padding > window.innerHeight) {
    top = window.innerHeight - dialogHeight - padding;
  }
  if (top < padding) {
    top = padding;
  }
  
  // Use portal to render outside the sidebar DOM hierarchy
  return createPortal(
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 z-[9999] bg-background/80 backdrop-blur-sm"
        onClick={handleBackdropClick}
      />
      {/* Dialog positioned near the click */}
      <div
        className="fixed z-[9999] w-80 rounded-lg border bg-background p-4 shadow-lg"
        style={{ 
          left: `${left}px`, 
          top: `${top}px`,
        }}
      >
        <h3 className="text-lg font-semibold mb-2">Delete Confirmation</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Are you sure you want to delete "{title}"?
        </p>
        <div className="flex items-center space-x-2 mb-4">
          <Checkbox 
            id="dontAskAgain" 
            checked={dontAskAgain} 
            onCheckedChange={setDontAskAgain} 
          />
          <Label htmlFor="dontAskAgain" className="text-sm">Don't ask again</Label>
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
          <Button variant="destructive" size="sm" onClick={handleConfirm}>Delete</Button>
        </div>
      </div>
    </>,
    document.body
  );
};

export default DeleteConfirmDialog;