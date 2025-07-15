import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardDescription, 
  CardContent, 
  CardFooter 
} from './ui/card';
import { Label } from './ui/label';
import { Switch } from './ui/switch';
import { Button } from './ui/button';
import { Checkbox } from './ui/checkbox';
import { FileText, RefreshCw, Info } from 'lucide-react';
import { 
  Tooltip, 
  TooltipContent, 
  TooltipProvider, 
  TooltipTrigger 
} from './ui/tooltip';

const RAGSettings = () => {
  const { 
    settings, 
    updateSettings, 
    documents, 
    fetchDocuments 
  } = useApp();
  
  const [isLoading, setIsLoading] = useState(false);
  
  // Ensure loading available documents on mount
  useEffect(() => {
    loadDocuments();
  }, []);
  
  const loadDocuments = async () => {
    setIsLoading(true);
    try {
      await fetchDocuments();
    } catch (error) {
      console.error("Error loading documents:", error);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Toggle RAG on/off
  const handleToggleRAG = (checked) => {
    updateSettings({ use_rag: checked });
    
    // Clear selected documents when disabling
    if (!checked) {
      updateSettings({ selectedDocuments: [] });
    }
  };
  
  // Toggle a document selection
  const toggleDocSelection = (docId) => {
    const currentSelected = settings.selectedDocuments || [];
    
    if (currentSelected.includes(docId)) {
      // Remove if already selected
      updateSettings({ 
        selectedDocuments: currentSelected.filter(id => id !== docId) 
      });
    } else {
      // Add if under limit (5 documents)
      if (currentSelected.length < 5) {
        updateSettings({ 
          selectedDocuments: [...currentSelected, docId] 
        });
      }
    }
  };
  useEffect(() => {
    console.log("Current RAG settings:", {
      enabled: settings.use_rag,
      selectedDocs: settings.selectedDocuments
    });
  }, [settings.use_rag, settings.selectedDocuments]);
  // Get count of selected documents
  const selectedCount = (settings.selectedDocuments || []).length;
  
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Document Context
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">
                  Document context allows the LLM to reference content from your uploaded files
                  when generating responses. Select up to 5 documents to include.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </CardTitle>
        <CardDescription>
          Enhance LLM responses with information from your documents
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Main RAG toggle */}
        <div className="flex items-center justify-between">
          <div>
            <Label htmlFor="use-rag" className="text-base font-medium">
              Enable Document Context
            </Label>
            <p className="text-sm text-muted-foreground mt-1">
              Use content from your documents to enhance LLM responses
            </p>
          </div>
          <Switch 
            id="use-rag" 
            checked={settings.use_rag || false}
            onCheckedChange={handleToggleRAG}
          />
        </div>
        
        {/* Document selection section - only shown when RAG is enabled */}
        {settings.use_rag && (
          <div className="p-4 border rounded-md mt-4 bg-muted/10">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium">
                Selected Documents ({selectedCount}/5)
              </h3>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={loadDocuments}
                disabled={isLoading}
              >
                <RefreshCw className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
                {isLoading ? 'Loading...' : 'Refresh'}
              </Button>
            </div>
            
            {/* Document list */}
            {documents?.file_list?.length > 0 ? (
              <div className="max-h-60 overflow-y-auto border rounded-md divide-y">
                {documents.file_list.map((doc) => {
                  const isSelected = (settings.selectedDocuments || []).includes(doc.id);
                  
                  return (
                    <div 
                      key={doc.id}
                      className={`flex items-center p-3 hover:bg-accent/10 ${
                        isSelected ? 'bg-primary/10' : ''
                      }`}
                    >
                      <Checkbox
                        id={`doc-${doc.id}`}
                        checked={isSelected}
                        onCheckedChange={() => toggleDocSelection(doc.id)}
                        disabled={!isSelected && selectedCount >= 5}
                        className="mr-3"
                      />
                      <div className="flex-1 min-w-0">
                        <label 
                          htmlFor={`doc-${doc.id}`}
                          className="flex items-center cursor-pointer"
                        >
                          <FileText className="h-4 w-4 mr-2 flex-shrink-0 text-primary" />
                          <div className="truncate">
                            <span className="font-medium truncate block">
                              {doc.filename}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {doc.file_type?.toUpperCase() || 'FILE'}
                              {doc.size_bytes && ` • ${(doc.size_bytes / 1024).toFixed(1)} KB`}
                            </span>
                          </div>
                        </label>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground border rounded-md">
                {isLoading ? (
                  <div className="flex justify-center items-center">
                    <RefreshCw className="h-5 w-5 animate-spin mr-2" />
                    <span>Loading documents...</span>
                  </div>
                ) : (
                  <>
                    <FileText className="h-10 w-10 mx-auto mb-2 opacity-50" />
                    <p>No documents available</p>
                    <p className="text-sm mt-1">
                      Upload files in the Documents tab first
                    </p>
                  </>
                )}
              </div>
            )}
            
            {/* Help text */}
            <p className="text-xs text-muted-foreground mt-3">
              The model will search for relevant information in these documents when generating responses.
            </p>
          </div>
        )}
      </CardContent>
      
      <CardFooter className="justify-between border-t pt-4">
        <div className="text-sm text-muted-foreground">
          {settings.use_rag 
            ? `${selectedCount} document${selectedCount !== 1 ? 's' : ''} selected` 
            : 'Document context is disabled'}
        </div>
        
        {settings.use_rag && selectedCount > 0 && (
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => updateSettings({ selectedDocuments: [] })}
          >
            Clear Selection
          </Button>
        )}
      </CardFooter>
    </Card>
  );
};

export default RAGSettings;