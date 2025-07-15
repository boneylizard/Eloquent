import React from 'react';
import { useApp } from '../contexts/AppContext';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Label } from './ui/label';
import { Switch } from './ui/switch';
import { Button } from './ui/button';
import { FileText, RefreshCw } from 'lucide-react';
import DocumentSelector from './DocumentSelector';

const DocumentSettings = () => {
  const { 
    settings, 
    updateSettings, 
    documents, 
    fetchDocuments 
  } = useApp();
  
  const handleUseRagToggle = (checked) => {
    updateSettings({ use_rag: checked });
    
    // If turning off RAG, clear selected documents
    if (!checked) {
      updateSettings({ selectedDocuments: [] });
    }
  };
  
  const handleDocumentSelection = (selectedDocs) => {
    updateSettings({ selectedDocuments: selectedDocs });
  };

  // Refresh documents list
  const handleRefresh = async () => {
    try {
      await fetchDocuments();
    } catch (err) {
      console.error("Error refreshing documents", err);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Document Integration</CardTitle>
        <CardDescription>
          Use documents to enhance LLM responses
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <Label htmlFor="use-rag" className="font-medium">Enable Document References</Label>
            <p className="text-sm text-muted-foreground">
              Use document content as context for responses
            </p>
          </div>
          <Switch 
            id="use-rag" 
            checked={settings.use_rag || false}
            onCheckedChange={handleUseRagToggle}
          />
        </div>
        
        {settings.use_rag && (
          <>
            <div className="space-y-2 pt-4 border-t">
              <div className="flex items-center justify-between mb-2">
                <Label>Selected Documents</Label>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleRefresh}
                >
                  <RefreshCw className="h-4 w-4 mr-1" /> Refresh
                </Button>
              </div>
              
              {/* Use the DocumentSelector component */}
              <DocumentSelector
                selectedDocs={settings.selectedDocuments || []}
                onChange={handleDocumentSelection}
                maxSelections={5}
              />
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default DocumentSettings;