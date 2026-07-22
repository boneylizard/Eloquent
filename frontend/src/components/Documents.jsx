import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Input } from './ui/input';
import { Checkbox } from './ui/checkbox';
import { Label } from './ui/label';
import { Switch } from './ui/switch';
import { Loader2, FileText, Trash2, RefreshCw, Upload, Eye, Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import DocumentPreview from './DocumentPreview';
import { getFileIcon, formatFileSize, validateFile } from '../utils/DocumentUtils';

const Documents = () => {
  const { PRIMARY_API_URL, documents, fetchDocuments, settings, updateSettings } = useApp();
  const [loading, setLoading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [error, setError] = useState(null);

  // Load documents on component mount
  useEffect(() => {
    refreshDocuments();
  }, []);

  const refreshDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      await fetchDocuments();
    } catch (err) {
      setError("Failed to fetch documents: " + err.message);
      console.error("Document fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(files);
  };

  const getFileEmoji = (filename) => {
    const ext = filename.split('.').pop().toLowerCase();
    switch(ext) {
      case 'pdf': return '📄';
      case 'doc':
      case 'docx': return '📝';
      case 'txt': return '📃';
      case 'csv': return '📊';
      case 'json': return '📋';
      case 'md': return '📑';
      default: return '📁';
    }
  };

  const uploadFiles = async () => {
    if (selectedFiles.length === 0) return;

    setUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      selectedFiles.forEach(file => formData.append('files', file));

      console.log(`Uploading ${selectedFiles.length} file(s):`, selectedFiles.map(f => f.name).join(', '));

      const response = await fetch(`${PRIMARY_API_URL}/document/upload-multiple`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      console.log("Upload response:", data);

      if (!response.ok) {
        throw new Error(data.detail || `Server returned ${response.status}`);
      }

      if (data.errors && data.errors.length > 0) {
        const errorMessages = data.errors.map(e => `${e.filename}: ${e.error}`).join('; ');
        setError(`Some files failed: ${errorMessages}`);
      }

      // Force refresh document list
      await fetchDocuments();
      setSelectedFiles([]);
      setUploadProgress(100);
    } catch (err) {
      console.error("Upload error:", err);
      setError("Upload failed: " + err.message);
    } finally {
      setUploading(false);
    }
  };

  const deleteDocument = async (docId) => {
    if (!confirm("Are you sure you want to delete this document? It will also be removed from document context.")) return;

    setDeleting(true);
    setError(null);

    try {
      const res = await fetch(`${PRIMARY_API_URL}/document/delete/${docId}`, {
        method: 'DELETE',
      });

      if (!res.ok) throw new Error(`Server returned ${res.status}`);

      // Drop the deleted doc from the context selection so we don't reference a ghost.
      const currentSelected = settings.selectedDocuments || [];
      if (currentSelected.includes(docId)) {
        updateSettings({ selectedDocuments: currentSelected.filter(id => id !== docId) });
      }

      await refreshDocuments();
    } catch (err) {
      setError("Failed to delete document: " + err.message);
      console.error("Document delete error:", err);
    } finally {
      setDeleting(false);
    }
  };

  const handleToggleRAG = (checked) => {
    updateSettings({ use_rag: checked, ...(!checked ? { ragAgentTools: false } : {}) });
    if (!checked) {
      updateSettings({ selectedDocuments: [] });
    }
  };

  const toggleDocSelection = (docId) => {
    const currentSelected = settings.selectedDocuments || [];
    if (currentSelected.includes(docId)) {
      updateSettings({ selectedDocuments: currentSelected.filter(id => id !== docId) });
    } else {
      updateSettings({ selectedDocuments: [...currentSelected, docId] });
    }
  };

  const handleSelectAll = () => {
    const allDocIds = (documents?.file_list || []).map(d => d.id);
    updateSettings({ selectedDocuments: allDocIds });
  };

  const handleClearSelection = () => {
    updateSettings({ selectedDocuments: [], ragAgentTools: false });
  };

  const docs = documents?.file_list || [];
  const selectedCount = (settings.selectedDocuments || []).length;

  return (
    <div className="container max-w-6xl mx-auto py-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Documents</h2>

        <div className="flex items-center space-x-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button onClick={refreshDocuments} variant="outline" disabled={loading}>
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  <span className="ml-2">Refresh</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Reload document list</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      <Tabs defaultValue="documents">
        <TabsList className="mb-4">
          <TabsTrigger value="documents">My Documents</TabsTrigger>
          <TabsTrigger value="upload">Upload Files</TabsTrigger>
        </TabsList>

        <TabsContent value="documents">
          {/* Document Context control — the single source of truth for RAG */}
          <Card className="mb-4">
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
                        When enabled, the model searches the checked documents for relevant
                        passages and weaves them into its reply.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
              <CardDescription>
                Let the model read your files when it answers.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="use-rag" className="text-base font-medium">
                    Enable Document Context
                  </Label>
                  <p className="text-sm text-muted-foreground mt-1">
                    Append matching passages from your checked files into the model's context.
                  </p>
                </div>
                <Switch
                  id="use-rag"
                  checked={settings.use_rag || false}
                  onCheckedChange={handleToggleRAG}
                />
              </div>

              {settings.use_rag && (
                <div className="space-y-4 pt-3 border-t">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-muted-foreground">
                      {selectedCount === 0
                        ? 'No documents checked — context is on but nothing to search.'
                        : `${selectedCount} document${selectedCount !== 1 ? 's' : ''} checked for context.`}
                    </div>
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 text-xs"
                        onClick={handleSelectAll}
                        disabled={docs.length === 0}
                      >
                        Check all
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 text-xs"
                        onClick={handleClearSelection}
                        disabled={selectedCount === 0}
                      >
                        Clear
                      </Button>
                    </div>
                  </div>

                  <div className="flex items-center justify-between gap-4 rounded-md border bg-muted/20 p-3">
                    <div>
                      <Label htmlFor="rag-agent-tools" className="text-sm font-medium">
                        Agent document search
                      </Label>
                      <p className="text-xs text-muted-foreground mt-1">
                        Compatible API models may search the checked documents as a tool. Other models keep using automatic document context.
                      </p>
                    </div>
                    <Switch
                      id="rag-agent-tools"
                      checked={settings.ragAgentTools === true}
                      onCheckedChange={(checked) => updateSettings({ ragAgentTools: checked === true })}
                      disabled={selectedCount === 0}
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Document library */}
          <Card>
            <CardHeader>
              <CardTitle>Available Documents</CardTitle>
            </CardHeader>
            <CardContent>
              {error && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded">
                  {error}
                </div>
              )}

              {loading ? (
                <div className="flex justify-center items-center p-8">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  <span className="ml-2">Loading documents...</span>
                </div>
              ) : (
                <>
                  {docs.length > 0 ? (
                    <div className="space-y-2">
                      <p className="text-xs text-muted-foreground px-1 pb-1">
                        {settings.use_rag
                          ? 'Tick the box beside a file to include it in document context. Use the trash icon to remove a file entirely.'
                          : 'Enable Document Context above to choose which files the model can read.'}
                      </p>
                      {docs.map((doc) => {
                        const isSelected = (settings.selectedDocuments || []).includes(doc.id);
                        return (
                          <div
                            key={doc.id}
                            className={`flex items-center justify-between p-3 border rounded hover:bg-accent/20 ${
                              isSelected ? 'bg-primary/10' : ''
                            }`}
                          >
                            <div className="flex items-center flex-1 min-w-0">
                              <Checkbox
                                checked={isSelected}
                                onCheckedChange={() => toggleDocSelection(doc.id)}
                                disabled={!settings.use_rag}
                                aria-label={`Include ${doc.filename} in document context`}
                                className="mr-3"
                              />
                              <span className="text-2xl mr-3 flex-shrink-0">{getFileEmoji(doc.filename)}</span>
                              <div className="min-w-0">
                                <div className="font-medium truncate">{doc.filename}</div>
                                <div className="text-sm text-muted-foreground">
                                  Added: {new Date(doc.upload_date).toLocaleString()}
                                </div>
                              </div>
                            </div>

                            <div className="flex space-x-2 flex-shrink-0 ml-2">
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Button size="sm" variant="outline">
                                      <Eye className="h-4 w-4" />
                                    </Button>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p>View document</p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>

                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Button
                                      size="sm"
                                      variant="destructive"
                                      onClick={() => deleteDocument(doc.id)}
                                      disabled={deleting}
                                    >
                                      <Trash2 className="h-4 w-4" />
                                    </Button>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p>Delete document</p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="text-center p-8 text-muted-foreground">
                      <FileText className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>No documents yet. Upload a file to give the model something to read.</p>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>Upload New Documents</CardTitle>
            </CardHeader>
            <CardContent>
              {error && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded">
                  {error}
                </div>
              )}

              <div className="space-y-4">
                <div className="border-2 border-dashed rounded-lg p-6 text-center hover:bg-accent/10 transition cursor-pointer">
                  <Input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    multiple
                    onChange={handleFileChange}
                    accept=".pdf,.doc,.docx,.txt,.md,.csv,.json,.py,.js,.html,.css"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                    <p className="mb-1">Click to select files</p>
                    <p className="text-sm text-muted-foreground">
                      Supports PDF, Word, text, markdown, CSV, code files
                    </p>
                  </label>
                </div>

                {selectedFiles.length > 0 && (
                  <div className="mt-4">
                    <h3 className="font-medium mb-2">Selected Files ({selectedFiles.length})</h3>
                    <div className="space-y-2 max-h-60 overflow-y-auto p-2 border rounded">
                      {selectedFiles.map((file, index) => (
                        <div key={index} className="flex items-center p-2 hover:bg-accent/10 rounded">
                          <span className="text-xl mr-2">{getFileEmoji(file.name)}</span>
                          <span className="flex-1 truncate">{file.name}</span>
                          <span className="text-sm text-muted-foreground">
                            {(file.size / 1024).toFixed(1)} KB
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {uploading && (
                  <div className="mt-4">
                    <div className="flex justify-between mb-1">
                      <span>Uploading {selectedFiles.length} file(s)...</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <Progress value={uploadProgress} className="h-2" />
                  </div>
                )}

                <div className="flex justify-end mt-4">
                  <Button
                    onClick={uploadFiles}
                    disabled={uploading || selectedFiles.length === 0}
                  >
                    {uploading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="mr-2 h-4 w-4" />
                        Upload {selectedFiles.length > 0 ? `(${selectedFiles.length})` : ''}
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Documents;
