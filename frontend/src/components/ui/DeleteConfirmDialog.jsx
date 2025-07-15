import React, { useState } from 'react';
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from './dialog';
import { Button } from './button';
import { Checkbox } from './checkbox';
import { Label } from './label';

const DeleteConfirmDialog = ({ isOpen, onClose, onConfirm, title }) => {
  const [dontAskAgain, setDontAskAgain] = useState(false);
  
  const handleConfirm = () => {
    // Pass the dontAskAgain value to the parent component
    onConfirm(dontAskAgain);
    onClose();
  };
  
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Delete Confirmation</DialogTitle>
        </DialogHeader>
        <div className="py-4">
          <p>Are you sure you want to delete "{title}"?</p>
          <div className="flex items-center space-x-2 mt-4">
            <Checkbox 
              id="dontAskAgain" 
              checked={dontAskAgain} 
              onCheckedChange={setDontAskAgain} 
            />
            <Label htmlFor="dontAskAgain">Don't ask again and automatically delete</Label>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button variant="destructive" onClick={handleConfirm}>Delete</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default DeleteConfirmDialog;