import React, { useState, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Label } from './ui/label';
import { Upload, X, Send, Image as ImageIcon } from 'lucide-react';

const ChatImageUploadButton = () => {
const {
  sendMessage,
  isGenerating,
  generateUniqueId,
  setMessages,
  userProfile,
  primaryModel,
  PRIMARY_API_URL,
  activeCharacter,
  settings,
  userCharacter,
} = useApp();

  // Component state
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [messageText, setMessageText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // File input ref
  const fileInputRef = useRef(null);

  // Allowed image types
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
  const maxFileSize = 10 * 1024 * 1024; // 10MB limit

  // Convert file to base64
  const fileToBase64 = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }, []);

  // Handle file selection
  const handleFileSelect = useCallback(async (file) => {
    if (!file) return;

    // Validate file type
    if (!allowedTypes.includes(file.type)) {
      alert('Please select a valid image file (JPEG, PNG, WebP, or GIF)');
      return;
    }

    // Validate file size
    if (file.size > maxFileSize) {
      alert('File size must be less than 10MB');
      return;
    }

    setIsProcessing(true);
    try {
      // Create preview URL
      const previewUrl = URL.createObjectURL(file);
      setImagePreviewUrl(previewUrl);
      
      // Convert to base64
      const base64Data = await fileToBase64(file);
      setSelectedImage({
        file,
        name: file.name,
        size: file.size,
        type: file.type,
        base64: base64Data
      });
      
      setIsDialogOpen(true);
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Error processing image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  }, [fileToBase64]);

  // Handle button click - open file picker
  const handleButtonClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Handle file input change
  const handleFileInputChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
    // Clear the input so the same file can be selected again
    e.target.value = '';
  }, [handleFileSelect]);

const handleSendWithImage = useCallback(async () => {
  if (!selectedImage || isGenerating) return;

  try {
    // Upload the image to get a permanent URL
    const formData = new FormData();
    formData.append('file', selectedImage.file);

    const uploadResponse = await fetch(`${PRIMARY_API_URL}/upload_avatar`, {
      method: 'POST',
      body: formData
    });

    if (!uploadResponse.ok) throw new Error('Failed to upload image');
    
    const uploadData = await uploadResponse.json();
    const imageUrl = uploadData.file_url;

    // Add user message with text (if any)
    if (messageText.trim()) {
      const userMsg = {
        id: generateUniqueId(),
        role: 'user',
        content: messageText.trim()
      };
      if (settings.multiRoleMode && userCharacter) {
        userMsg.characterId = userCharacter.id;
        userMsg.characterName = userCharacter.name;
        userMsg.avatar = userCharacter.avatar;
      }
      setMessages(prev => [...prev, userMsg]);
    }

    // Add the image message (same format as SimpleChatImageButton)
    const imageMessage = {
      id: generateUniqueId(),
      role: 'bot',
      type: 'image',
      content: 'Uploaded image',
      imagePath: imageUrl,
      prompt: 'Uploaded image',
      width: 512,
      height: 512,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, imageMessage]);

    // If there's text, call vision API for analysis
    if (messageText.trim()) {
      const systemPrompt = activeCharacter 
        ? `System: You are ${activeCharacter.name}. ${activeCharacter.description}\n\n${activeCharacter.model_instructions}`
        : 'System: You are a helpful AI assistant.';
        
      const fullPrompt = `${systemPrompt}\n\nHuman: ${messageText.trim()}`;        
      const response = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: fullPrompt,
          model_name: primaryModel,
          image_base64: selectedImage.base64.split(',')[1],
          image_type: selectedImage.type,
          temperature: 0.7,
          max_tokens: 1024,
          userProfile: { id: userProfile?.id ?? 'anonymous' }
        })
      });

      if (response.ok) {
        const result = await response.json();
        const botMsg = {
          id: generateUniqueId(),
          role: 'bot',
          content: result.text || 'No response from vision model',
          modelId: 'primary'
        };
        setMessages(prev => [...prev, botMsg]);
      }
    }

    // Reset state and close dialog
    handleCloseDialog();
    
  } catch (error) {
    console.error('Error processing image:', error);
    alert(`Error processing image: ${error.message}`);
  }
}, [selectedImage, messageText, isGenerating, generateUniqueId, setMessages, primaryModel, userProfile, PRIMARY_API_URL, settings, userCharacter]);

  // Handle closing the dialog and cleanup
  const handleCloseDialog = useCallback(() => {
    setIsDialogOpen(false);
    setSelectedImage(null);
    setMessageText('');
    
    // Cleanup preview URL to prevent memory leaks
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
      setImagePreviewUrl(null);
    }
  }, [imagePreviewUrl]);

  // Format file size for display
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <>
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={allowedTypes.join(',')}
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />

      {/* Upload button */}
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 rounded-full p-0"
        title="Upload Image"
        onClick={handleButtonClick}
        disabled={isProcessing || isGenerating}
      >
        {isProcessing ? (
          <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
        ) : (
          <Upload className="h-4 w-4" />
        )}
      </Button>

      {/* Image preview and message dialog */}
      {isDialogOpen && createPortal(
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div
            className="relative w-full max-w-2xl bg-background rounded-lg p-6 shadow-xl max-h-[90vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between border-b pb-3 mb-4">
              <div>
                <h3 className="text-lg font-semibold">Upload Image</h3>
                <p className="text-sm text-muted-foreground">Add an image to your message</p>
              </div>
              <Button variant="ghost" size="icon" className="-mt-1 -mr-2" onClick={handleCloseDialog}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="space-y-4 overflow-y-auto flex-1">
              {/* Image preview */}
              {imagePreviewUrl && (
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Image Preview</Label>
                  <div className="border rounded-lg p-4 bg-muted/50">
                    <img
                      src={imagePreviewUrl}
                      alt="Preview"
                      className="max-w-full max-h-64 mx-auto rounded object-contain"
                    />
                    <div className="mt-2 text-xs text-muted-foreground text-center">
                      {selectedImage?.name} • {formatFileSize(selectedImage?.size || 0)}
                    </div>
                  </div>
                </div>
              )}

              {/* Message text input */}
              <div className="space-y-2">
                <Label htmlFor="message-text" className="text-sm font-medium">
                  Message (optional)
                </Label>
                <Textarea
                  id="message-text"
                  placeholder="Add a message about this image..."
                  value={messageText}
                  onChange={(e) => setMessageText(e.target.value)}
                  rows={3}
                  className="resize-none"
                />
              </div>

              {/* Info note */}
              <div className="text-xs text-muted-foreground bg-muted/50 p-3 rounded">
                <strong>Note:</strong> Vision support is currently in development. 
                The image will be uploaded but may not be processed by the AI model yet.
              </div>
            </div>

            {/* Footer buttons */}
            <div className="flex justify-end gap-3 mt-4 pt-4 border-t">
              <Button variant="outline" onClick={handleCloseDialog}>
                Cancel
              </Button>
              <Button 
                onClick={handleSendWithImage}
                disabled={!selectedImage || isGenerating}
                className="min-w-[100px]"
              >
                <Send className="w-4 h-4 mr-2" />
                Send
              </Button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
};

export default ChatImageUploadButton;
