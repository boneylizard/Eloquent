import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Button } from './ui/button';
import { FileText, X, Check, Search } from 'lucide-react';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { getFileIcon } from '../utils/DocumentUtils';

const DocumentSelector = ({ selectedDocs = [], onChange }) => {
  const { documents, fetchDocuments } = useApp();
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Refresh document list when dialog opens
  useEffect(() => {
    if (isOpen) {
      refreshDocuments();
    }
  }, [isOpen]);

  const refreshDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      await fetchDocuments();
    } catch (err) {
      setError("Failed to load documents");
      console.error("Document fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  const toggleDocument = (docId) => {
    if (selectedDocs.includes(docId)) {
      onChange(selectedDocs.filter(id => id !== docId));
    } else {
      onChange([...selectedDocs, docId]);
    }
  };

  const clearSelections = () => {
    onChange([]);
  };

  const selectAllDocuments = () => {
    const available = filteredDocuments.map(d => d.id);
    const current = selectedDocs.filter(id => available.includes(id));
    const newOnes = available.filter(id => !current.includes(id));
    onChange([...current, ...newOnes]);
  };

  // Filter documents by search query
  const filteredDocuments = documents?.file_list?.filter(doc => {
    if (!searchQuery.trim()) return true;
    return doc.filename.toLowerCase().includes(searchQuery.toLowerCase());
  }) || [];

  return (
    <div>
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogTrigger asChild>
          <Button 
            variant="outline"
            className="flex items-center justify-between w-full text-left"
          >
            <div className="flex items-center">
              <FileText className="h-4 w-4 mr-2" />
              <span>
                {selectedDocs.length === 0 ? (
                  "Select documents to reference"
                ) : (
                  `${selectedDocs.length} document${selectedDocs.length > 1 ? 's' : ''} selected`
                )}
              </span>
            </div>
            {selectedDocs.length > 0 && (
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-6 w-6 p-0" 
                onClick={(e) => {
                  e.stopPropagation();
                  clearSelections();
                }}
              >
                <X className="h-3 w-3" />
              </Button>
            )}
          </Button>
        </DialogTrigger>
        
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Select Documents</DialogTitle>
          </DialogHeader>
          
          <div className="mt-4 space-y-4">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search documents..."
                className="pl-10"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            {error && (
              <div className="text-sm text-red-500">
                {error}
                <Button 
                  variant="link" 
                  size="sm" 
                  className="p-0 h-auto ml-2" 
                  onClick={refreshDocuments}
                >
                  Retry
                </Button>
              </div>
            )}
            
            {loading ? (
              <div className="text-center py-4 text-muted-foreground">
                Loading documents...
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between mb-2">
                  <div className="text-sm text-muted-foreground">
                    {selectedDocs.length} document{selectedDocs.length !== 1 ? 's' : ''} selected
                  </div>
                  <div className="flex gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={selectAllDocuments}
                      disabled={filteredDocuments.length === 0}
                    >
                      Select All
                    </Button>
                    {selectedDocs.length > 0 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={clearSelections}
                      >
                        Clear
                      </Button>
                    )}
                  </div>
                </div>
                
                {filteredDocuments.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    {searchQuery ? (
                      `No documents matching "${searchQuery}"`
                    ) : (
                      "No documents available. Upload some in the Documents tab!"
                    )}
                  </div>
                ) : (
                  <ScrollArea className="h-[300px] border rounded-md">
                    <div className="p-2 space-y-1">
                      {filteredDocuments.map((doc) => {
                        const isSelected = selectedDocs.includes(doc.id);
                        
                        return (
                          <div 
                            key={doc.id}
                            className={`flex items-center p-2 rounded-md cursor-pointer ${
                              isSelected ? 'bg-primary/10 hover:bg-primary/20' : 'hover:bg-accent/20'
                            }`}
                            onClick={() => toggleDocument(doc.id)}
                          >
                            <div className="mr-2">
                              {isSelected ? (
                                <div className="h-5 w-5 rounded-full bg-primary flex items-center justify-center">
                                  <Check className="h-3 w-3 text-white" />
                                </div>
                              ) : (
                                <div className="h-5 w-5 rounded-full border border-muted-foreground" />
                              )}
                            </div>
                            <span className="text-xl mr-2">{getFileIcon(doc.filename)}</span>
                            <div className="flex-1 truncate">
                              <div className="font-medium text-sm leading-snug truncate">
                                {doc.filename}
                              </div>
                              <div className="text-xs text-muted-foreground">
                                {doc.file_type?.toUpperCase() || 'UNKNOWN'} 
                                {doc.size_bytes && ` • ${(doc.size_bytes / 1024).toFixed(1)} KB`}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </ScrollArea>
                )}
                
                <div className="flex justify-end space-x-2 mt-2">
                  <Button 
                    variant="outline" 
                    onClick={() => setIsOpen(false)}
                  >
                    Done
                  </Button>
                </div>
              </>
            )}
          </div>
        </DialogContent>
      </Dialog>
      
      {/* Show selected documents summary */}
      {selectedDocs.length > 0 && (
        <div className="mt-2">
          <div className="text-sm font-medium mb-1">Selected Documents:</div>
          <div className="flex flex-wrap gap-1">
            {selectedDocs.map((docId) => {
              const doc = documents?.file_list?.find(d => d.id === docId);
              if (!doc) return null;
              
              return (
                <div 
                  key={docId}
                  className="inline-flex items-center bg-accent/20 rounded-full px-2 py-1 text-xs"
                >
                  <span className="mr-1">{getFileIcon(doc.filename)}</span>
                  <span className="truncate max-w-[120px]">{doc.filename}</span>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="h-4 w-4 p-0 ml-1" 
                    onClick={() => toggleDocument(docId)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentSelector;