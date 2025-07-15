import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from './ui/card';
import { Loader2, ArrowLeft, Download } from 'lucide-react';
import { getFileIcon, formatFileSize } from '../utils/DocumentUtils'; // Ensure this import path is correct

const DocumentPreview = ({ documentId, onBack }) => {
  const { getDocumentContent, documents } = useApp();
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [document, setDocument] = useState(null);
  
  useEffect(() => {
    const loadDocument = async () => {
      if (!documentId) return;
      
      // Find document metadata
      const docMeta = documents?.file_list?.find(d => d.id === documentId);
      if (docMeta) {
        setDocument(docMeta);
      }
      
      // Load content
      setLoading(true);
      setError(null);
      
      try {
        const docContent = await getDocumentContent(documentId);
        setContent(docContent);
      } catch (err) {
        setError(`Failed to load document content: ${err.message}`);
        console.error('Document preview error:', err);
      } finally {
        setLoading(false);
      }
    };
    
    loadDocument();
  }, [documentId, getDocumentContent, documents]);
  
  // Helper function to determine what content display to use
  const getContentDisplay = () => {
    if (!content) return null;
    
    // For code files, use syntax highlighting
    const extension = document?.filename?.split('.').pop()?.toLowerCase();
    const isCode = ['py', 'js', 'html', 'css', 'json'].includes(extension);
    
    if (isCode) {
      return (
        <pre className="whitespace-pre-wrap p-4 text-sm border rounded bg-slate-50 font-mono overflow-x-auto">
          {content}
        </pre>
      );
    }
    
    // For CSV, try to display as table
    if (extension === 'csv' && content.includes(',')) {
      try {
        const lines = content.trim().split('\n');
        const headers = lines[0].split(',');
        
        return (
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse border border-gray-300">
              <thead>
                <tr className="bg-gray-100">
                  {headers.map((header, i) => (
                    <th key={i} className="border border-gray-300 px-4 py-2 text-left">
                      {header.trim()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {lines.slice(1).map((line, i) => (
                  <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {line.split(',').map((cell, j) => (
                      <td key={j} className="border border-gray-300 px-4 py-2">
                        {cell.trim()}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      } catch (err) {
        // If table parsing fails, fall back to text display
        console.warn('Failed to parse CSV as table, using text display');
      }
    }
    
    // Default text display
    return (
      <div className="whitespace-pre-wrap p-4 text-sm border rounded bg-slate-50 overflow-x-auto">
        {content}
      </div>
    );
  };
  
  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center">
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onBack}
            className="mr-2"
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <CardTitle className="flex items-center">
            {document && (
              <>
                <span className="text-2xl mr-2">{getFileIcon(document.filename)}</span>
                <span>{document.filename}</span>
              </>
            )}
          </CardTitle>
        </div>
        
        {document && (
          <div className="text-sm text-muted-foreground">
            {document.file_type?.toUpperCase()} • {formatFileSize(document.size_bytes)}
          </div>
        )}
      </CardHeader>
      
      <CardContent className="max-h-[70vh] overflow-y-auto">
        {loading ? (
          <div className="flex justify-center items-center p-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <span className="ml-2">Loading document content...</span>
          </div>
        ) : error ? (
          <div className="p-4 bg-red-50 border border-red-200 text-red-700 rounded">
            {error}
          </div>
        ) : (
          getContentDisplay()
        )}
      </CardContent>
      
      <CardFooter className="flex justify-between border-t pt-4">
        <div className="text-sm text-muted-foreground">
          {document && `Uploaded on ${new Date(document.upload_date).toLocaleString()}`}
        </div>
        
        <Button variant="outline" size="sm" disabled={!document}>
          <Download className="h-4 w-4 mr-2" />
          Download
        </Button>
      </CardFooter>
    </Card>
  );
};

export default DocumentPreview;