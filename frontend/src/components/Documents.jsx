import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Loader2, FileText, Trash2, RefreshCw, Upload, Eye } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import DocumentPreview from './DocumentPreview';
import { getFileIcon, formatFileSize, validateFile } from '../utils/DocumentUtils';

const Documents = () => {
  const { PRIMARY_API_URL, documents, fetchDocuments } = useApp();
  const [loading, setLoading] = useState(false);
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

  const getFileIcon = (filename) => {
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
    
    try {
      // Upload just the first file as a test
      const file = selectedFiles[0];
      const formData = new FormData();
      formData.append('file', file);
      
      console.log("Uploading file:", file.name);
      
      const response = await fetch(`${PRIMARY_API_URL}/document/upload`, {
        method: 'POST',
        body: formData,
      });
      
      console.log("Upload response status:", response.status);
      
      const data = await response.json();
      console.log("Upload response data:", data);
      
      // Force refresh document list
      await fetchDocuments();
      setSelectedFiles([]);
    } catch (err) {
      console.error("Upload error:", err);
      setError("Upload failed: " + err.message);
    } finally {
      setUploading(false);
    }
  };

  const deleteDocument = async (docId) => {
    if (!confirm("Are you sure you want to delete this document?")) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch(`${PRIMARY_API_URL}/document/delete/${docId}`, {
        method: 'DELETE',
      });
      
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      
      await refreshDocuments();
    } catch (err) {
      setError("Failed to delete document: " + err.message);
      console.error("Document delete error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container max-w-6xl mx-auto py-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Document Manager</h2>
        
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
      
      <Tabs defaultValue="documents">
        <TabsList className="mb-4">
          <TabsTrigger value="documents">My Documents</TabsTrigger>
          <TabsTrigger value="upload">Upload Files</TabsTrigger>
        </TabsList>
        
        <TabsContent value="documents">
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
                  {documents?.file_list?.length > 0 ? (
                    <div className="space-y-2">
                      {documents.file_list.map((doc) => (
                        <div 
                          key={doc.id} 
                          className="flex items-center justify-between p-3 border rounded hover:bg-accent/20"
                        >
                          <div className="flex items-center">
                            <span className="text-2xl mr-3">{getFileIcon(doc.filename)}</span>
                            <div>
                              <div className="font-medium">{doc.filename}</div>
                              <div className="text-sm text-muted-foreground">
                                Added: {new Date(doc.upload_date).toLocaleString()}
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex space-x-2">
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
                      ))}
                    </div>
                  ) : (
                    <div className="text-center p-8 text-muted-foreground">
                      <FileText className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>No documents available. Upload files to use with your LLM!</p>
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
                          <span className="text-xl mr-2">{getFileIcon(file.name)}</span>
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
                      <span>Uploading...</span>
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