import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Lock } from 'lucide-react';

const LoginOverlay = ({ isOpen, onLogin }) => {
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!password.trim()) {
            setError('Password is required');
            return;
        }
        onLogin(password);
        setPassword('');
        setError('');
    };

    return (
        <Dialog open={isOpen} onOpenChange={() => { }}>
            <DialogContent className="sm:max-w-md [&>button]:hidden">
                {/* [&>button]:hidden hides the close X button to force login */}
                <DialogHeader>
                    <div className="mx-auto bg-primary/10 p-3 rounded-full w-fit mb-2">
                        <Lock className="h-6 w-6 text-primary" />
                    </div>
                    <DialogTitle className="text-center">Authentication Required</DialogTitle>
                    <DialogDescription className="text-center">
                        Please enter the remote access password to continue.
                    </DialogDescription>
                </DialogHeader>

                <form onSubmit={handleSubmit} className="space-y-4 py-4">
                    <div className="space-y-2">
                        <Input
                            type="password"
                            placeholder="Password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="text-center text-lg"
                            autoFocus
                        />
                        {error && <p className="text-sm text-destructive text-center">{error}</p>}
                    </div>

                    <DialogFooter className="sm:justify-center">
                        <Button type="submit" className="w-full sm:w-auto min-w-[150px]">
                            Unlock
                        </Button>
                    </DialogFooter>
                </form>
            </DialogContent>
        </Dialog>
    );
};

export default LoginOverlay;
