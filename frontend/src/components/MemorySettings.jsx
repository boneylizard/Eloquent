// --- MemorySettings.jsx - New Component for Memory Control ---

import React from 'react';
import { useMemory } from '../contexts/MemoryContext';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Slider } from './ui/slider';
import { Button } from './ui/button';
import { Trash2, RefreshCw, HardDrive } from 'lucide-react';
import { 
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from './ui/alert-dialog';
import { useApp } from '../contexts/AppContext';

const MemorySettings = () => {
  const { 
    memorySettings, 
    updateMemorySettings,
    memoryOptIn,
    toggleMemoryOptIn,
    explicitMemoryOnly,
    toggleExplicitMemoryOnly,
    resetMemories
  } = useMemory();
  const { BACKEND } = useApp();
  const [isPurgeDialogOpen, setIsPurgeDialogOpen] = React.useState(false);
  const [isCurating, setIsCurating] = React.useState(false);
  const [isPurging, setIsPurging] = React.useState(false);
  const [memoryStats, setMemoryStats] = React.useState(null);
  const [initializeMemories, setInitializeMemories] = React.useState(false);
  
  const fetchMemoryStats = async () => {
    try {
      const response = await fetch(`${BACKEND}/memory/stats`);
      if (response.ok) {
        const data = await response.json();
        setMemoryStats(data.stats);
      }
    } catch (error) {
      console.error('Error fetching memory stats:', error);
    }
  };
  
  React.useEffect(() => {
    fetchMemoryStats();
  }, []);
  
  const handleImportanceThresholdChange = (values) => {
    updateMemorySettings({
      ...memorySettings,
      memoryImportanceThreshold: values[0]
    });
  };
  
  const handleMaxMemoriesChange = (values) => {
    updateMemorySettings({
      ...memorySettings,
      maxMemoriesPerQuery: Math.round(values[0])
    });
  };
  
  const handleCurateMemories = async () => {
    setIsCurating(true);
    try {
      const response = await fetch(`${BACKEND}/memory/curate`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Memory curation result:', result);
        fetchMemoryStats();
      }
    } catch (error) {
      console.error('Error curating memories:', error);
    } finally {
      setIsCurating(false);
    }
  };
  
  const handlePurgeMemories = async () => {
    setIsPurging(true);
    try {
      const response = await fetch(`${BACKEND}/memory/purge`, {
        method: 'POST'
      });
      
      if (response.ok) {
        await response.json();
        resetMemories();
        fetchMemoryStats();
      }
    } catch (error) {
      console.error('Error purging memories:', error);
    } finally {
      setIsPurging(false);
      setIsPurgeDialogOpen(false);
    }
  };
  
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Memory System</CardTitle>
          <CardDescription>
            Control how the AI remembers information about you
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="memory-opt-in">Memory Enabled</Label>
              <div className="text-sm text-muted-foreground">
                Allow AI to use memories in conversations
              </div>
            </div>
            <Switch 
              id="memory-opt-in"
              checked={memoryOptIn}
              onCheckedChange={toggleMemoryOptIn}
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="explicit-memory-only">Explicit Memory Only</Label>
              <div className="text-sm text-muted-foreground">
                Only create memories when explicitly requested
              </div>
            </div>
            <Switch 
              id="explicit-memory-only"
              checked={explicitMemoryOnly}
              onCheckedChange={toggleExplicitMemoryOnly}
            />
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label htmlFor="importance-threshold">Importance Threshold</Label>
              <span className="text-sm text-muted-foreground">
                {memorySettings.memoryImportanceThreshold.toFixed(1)}
              </span>
            </div>
            <Slider
              id="importance-threshold"
              min={0.1}
              max={1.0}
              step={0.1}
              value={[memorySettings.memoryImportanceThreshold]}
              onValueChange={handleImportanceThresholdChange}
              disabled={!memoryOptIn}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>More Memories</span>
              <span>Higher Quality</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label htmlFor="max-memories">Max Memories Per Query</Label>
              <span className="text-sm text-muted-foreground">
                {memorySettings.maxMemoriesPerQuery}
              </span>
            </div>
            <Slider
              id="max-memories"
              min={1}
              max={10}
              step={1}
              value={[memorySettings.maxMemoriesPerQuery]}
              onValueChange={handleMaxMemoriesChange}
              disabled={!memoryOptIn}
            />
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Memory Management</CardTitle>
          <CardDescription>
            Clean up and organize stored memories
          </CardDescription>
        </CardHeader>
        <CardContent>
          {memoryStats && (
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="p-4 rounded-lg border bg-card text-card-foreground shadow-sm">
                <div className="text-2xl font-bold">{memoryStats.total_memories}</div>
                <div className="text-sm text-muted-foreground">Total Memories</div>
              </div>
              
              <div className="p-4 rounded-lg border bg-card text-card-foreground shadow-sm">
                <div className="text-2xl font-bold">
                  {memoryStats.by_importance?.high || 0}
                </div>
                <div className="text-sm text-muted-foreground">High Importance</div>
              </div>
            </div>
          )}
          
          <div className="space-y-4">
            <Button 
              variant="outline" 
              className="w-full justify-start" 
              onClick={handleCurateMemories}
              disabled={isCurating}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              {isCurating ? "Cleaning Up..." : "Clean Up Memories"}
              <span className="ml-auto text-xs text-muted-foreground">
                Removes duplicates & improves quality
              </span>
            </Button>
            
            <Button 
              variant="outline" 
              className="w-full justify-start text-destructive" 
              onClick={() => setIsPurgeDialogOpen(true)}
              disabled={isPurging}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              {isPurging ? "Deleting..." : "Purge All Memories"}
              <span className="ml-auto text-xs text-muted-foreground">
                Permanently deletes all memories
              </span>
            </Button>
            
            <Button 
              variant="outline" 
              className="w-full justify-start" 
              onClick={fetchMemoryStats}
            >
              <HardDrive className="mr-2 h-4 w-4" />
              Refresh Memory Stats
            </Button>
          </div>
        </CardContent>
      </Card>
      
      <AlertDialog open={isPurgeDialogOpen} onOpenChange={setIsPurgeDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete all your saved memories. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handlePurgeMemories} className="bg-destructive text-destructive-foreground">
              {isPurging ? "Deleting..." : "Yes, Delete All"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default MemorySettings;
