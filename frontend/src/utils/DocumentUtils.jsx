import React from 'react';

/**
 * Format a file size in bytes to a human-readable string
 * @param {number} bytes - Size in bytes
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} - Formatted size string
 */
export const formatFileSize = (bytes, decimals = 2) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
};

/**
 * Get an appropriate icon for a file based on its extension
 * @param {string} filename - Name of the file
 * @returns {string} - Icon emoji for the file type
 */
export const getFileIcon = (filename) => {
  const ext = filename.split('.').pop().toLowerCase();
  switch(ext) {
    case 'pdf': return '📄';
    case 'doc':
    case 'docx': return '📝';
    case 'txt': return '📃';
    case 'csv': return '📊';
    case 'json': return '📋';
    case 'md': return '📑';
    case 'py': return '🐍';
    case 'js': return '📜';
    case 'html': return '🌐';
    case 'css': return '🎨';
    default: return '📁';
  }
};

/**
 * Validates file before upload
 * @param {File} file - The file to validate
 * @returns {Object} - Validation result {valid: boolean, message: string}
 */
export const validateFile = (file) => {
  // Check file size (limit to 10MB)
  const MAX_SIZE = 10 * 1024 * 1024; // 10MB
  if (file.size > MAX_SIZE) {
    return {
      valid: false,
      message: `File size exceeds 10MB limit (${formatFileSize(file.size)})`
    };
  }
  
  // Check file type
  const ext = file.name.split('.').pop().toLowerCase();
  const supportedTypes = [
    'pdf', 'doc', 'docx', 'txt', 'md', 
    'csv', 'json', 'py', 'js', 'html', 'css'
  ];
  
  if (!supportedTypes.includes(ext)) {
    return {
      valid: false,
      message: `Unsupported file type: .${ext}`
    };
  }
  
  return { valid: true, message: 'File is valid' };
};

/**
 * Component to render document selection list for chat use
 */
export const DocumentSelector = ({ documents, selectedDocs, onSelectionChange }) => {
  if (!documents || !documents.file_list || documents.file_list.length === 0) {
    return (
      <div className="text-center p-4 text-muted-foreground">
        No documents available. Upload documents in the Documents tab.
      </div>
    );
  }
  
  return (
    <div className="max-h-60 overflow-y-auto border rounded">
      {documents.file_list.map((doc) => (
        <div 
          key={doc.id}
          className="flex items-center p-2 hover:bg-accent/10"
        >
          <input
            type="checkbox"
            id={`doc-${doc.id}`}
            checked={selectedDocs?.includes(doc.id)}
            onChange={(e) => {
              if (e.target.checked) {
                onSelectionChange([...selectedDocs, doc.id]);
              } else {
                onSelectionChange(selectedDocs.filter(id => id !== doc.id));
              }
            }}
            className="mr-2"
          />
          <label htmlFor={`doc-${doc.id}`} className="flex items-center cursor-pointer flex-1">
            <span className="text-xl mr-2">{getFileIcon(doc.filename)}</span>
            <span className="truncate">{doc.filename}</span>
          </label>
        </div>
      ))}
    </div>
  );
};

/**
 * Component for displaying a document preview
 */
export const DocumentPreview = ({ documentId, fetchContent, className }) => {
  const [content, setContent] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  
  React.useEffect(() => {
    const loadContent = async () => {
      if (!documentId) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const docContent = await fetchContent(documentId);
        setContent(docContent);
      } catch (err) {
        setError(`Failed to load document: ${err.message}`);
        console.error('Document preview error:', err);
      } finally {
        setLoading(false);
      }
    };
    
    loadContent();
  }, [documentId, fetchContent]);
  
  if (loading) {
    return (
      <div className={`flex justify-center items-center p-4 ${className}`}>
        <div className="animate-spin h-6 w-6 border-2 border-primary rounded-full border-t-transparent"></div>
        <span className="ml-2">Loading document...</span>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className={`p-4 bg-red-50 text-red-700 rounded ${className}`}>
        {error}
      </div>
    );
  }
  
  if (!content) {
    return (
      <div className={`p-4 text-center text-muted-foreground ${className}`}>
        Select a document to preview
      </div>
    );
  }
  
  return (
    <pre className={`whitespace-pre-wrap p-4 text-sm border rounded bg-slate-50 max-h-96 overflow-y-auto ${className}`}>
      {content}
    </pre>
  );
};

export default {
  formatFileSize,
  getFileIcon,
  validateFile,
  DocumentSelector,
  DocumentPreview
};