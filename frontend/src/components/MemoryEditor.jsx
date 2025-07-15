import React, { useState, useEffect, useCallback } from 'react';
import { useMemory } from '../contexts/MemoryContext';
import { useApp } from '../contexts/AppContext';
import ProfileSelector from './ProfileSelector';
import './MemoryEditor.css';
import { Input } from './ui/input';

const MEMORY_CATEGORIES = [
  'personal_info',
  'preferences',
  'interests',
  'facts',
  'skills',
  'opinions',
  'experiences',
  'other'
];

const MemoryEditor = ({ onClose }) => {
  const { 
    userProfile,
    activeProfileId,
    updateUserProfile
  } = useMemory();
  
  // Get MEMORY_API_URL from AppContext
  const { MEMORY_API_URL } = useApp();
  
  // Backend-only memory state
  const [memories, setMemories] = useState([]);
  const [isLoadingMemories, setIsLoadingMemories] = useState(false);
  const [memoryError, setMemoryError] = useState(null);
  
  const [isUploading, setIsUploading] = useState(false);
  const [activeView, setActiveView] = useState('memories');
  const [editingMemory, setEditingMemory] = useState(null);
  const [newMemory, setNewMemory] = useState({
    content: '',
    category: 'other',
    importance: 0.5
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [profileData, setProfileData] = useState({ ...userProfile });


  // Fetch memories from backend
  const fetchMemories = useCallback(async () => {
    if (!activeProfileId) {
      setMemories([]);
      return;
    }

    setIsLoadingMemories(true);
    setMemoryError(null);
    
    try {
      const response = await fetch(`${MEMORY_API_URL}/memory/get_all?user_id=${activeProfileId}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch memories: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setMemories(data.memories || []);
        console.log(`Loaded ${data.memories?.length || 0} memories for profile ${activeProfileId}`);
      } else {
        throw new Error(data.detail || 'Failed to fetch memories');
      }
    } catch (error) {
      console.error('Error fetching memories:', error);
      setMemoryError(error.message);
      setMemories([]);
    } finally {
      setIsLoadingMemories(false);
    }
  }, [activeProfileId, MEMORY_API_URL]);

  // Load memories when profile changes
  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  // Add memory via backend API
  const addMemory = useCallback(async (memoryData) => {
    if (!activeProfileId) {
      alert('No active profile selected');
      return false;
    }

    try {
      const response = await fetch(`${MEMORY_API_URL}/memory/add`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...memoryData,
          user_id: activeProfileId
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      
      if (result.status === 'success') {
        // Refresh memories from backend
        await fetchMemories();
        return true;
      } else {
        throw new Error(result.detail || 'Memory creation failed');
      }
    } catch (error) {
      console.error('Error adding memory:', error);
      alert(`Failed to add memory: ${error.message}`);
      return false;
    }
  }, [activeProfileId, MEMORY_API_URL, fetchMemories]);

  // Delete memory via backend API
  const deleteMemory = useCallback(async (memoryContent) => {
    if (!activeProfileId) {
      alert('No active profile selected');
      return false;
    }

    if (!window.confirm('Are you sure you want to delete this memory?')) {
      return false;
    }

    try {
      // Note: Using memory content for deletion since backend expects content_to_delete
      const response = await fetch(`${MEMORY_API_URL}/memory/delete_memory_for_user`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: activeProfileId,
          content_to_delete: memoryContent
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // Refresh memories from backend
        await fetchMemories();
        return true;
      } else {
        throw new Error('Memory deletion failed');
      }
    } catch (error) {
      console.error('Error deleting memory:', error);
      alert(`Failed to delete memory: ${error.message}`);
      return false;
    }
  }, [activeProfileId, MEMORY_API_URL, fetchMemories]);

  // Reset all memories via backend API
  const resetMemories = useCallback(async () => {
    if (!activeProfileId) {
      alert('No active profile selected');
      return;
    }

    if (!window.confirm('Are you sure you want to reset all memories for this profile? This cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(`${MEMORY_API_URL}/memory/clear?user_id=${activeProfileId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      
      if (result.status === 'success') {
        alert(`Successfully cleared ${result.cleared_memories || 0} memories`);
        // Refresh memories from backend
        await fetchMemories();
      } else {
        throw new Error(result.detail || 'Memory reset failed');
      }
    } catch (error) {
      console.error('Error resetting memories:', error);
      alert(`Failed to reset memories: ${error.message}`);
    }
  }, [activeProfileId, MEMORY_API_URL, fetchMemories]);

  const handleUserAvatarUpload = useCallback(async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    setIsUploading(true);

    try {
      console.log("🧠 Uploading user avatar...");
      const uploadUrl = `${MEMORY_API_URL}/upload_avatar`;

      const response = await fetch(uploadUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown server error' }));
        throw new Error(`Avatar upload failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const result = await response.json();

      if (result.status === 'success' && result.file_url) {
        console.log("🧠 User avatar uploaded successfully. URL:", result.file_url);
        updateUserProfile({ avatar: result.file_url });
        alert("Avatar updated successfully!");
      } else {
        throw new Error(result.detail || "Backend indicated upload failure.");
      }

    } catch (error) {
      console.error("Error uploading user avatar:", error);
      alert(`Avatar upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
      e.target.value = null;
    }
  }, [updateUserProfile, MEMORY_API_URL]);

  // Update profile data when userProfile changes
  useEffect(() => {
    setProfileData({ ...userProfile });
  }, [userProfile]);
  
  // Filter memories based on search and category
  const filteredMemories = memories.filter(memory => {
    const matchesSearch = searchQuery === '' || 
      memory.content.toLowerCase().includes(searchQuery.toLowerCase());
      
    const matchesCategory = categoryFilter === '' || 
      memory.category === categoryFilter;
      
    return matchesSearch && matchesCategory;
  });
  
  // Sort memories by importance and recency
  const sortedMemories = [...filteredMemories].sort((a, b) => {
    if (b.importance !== a.importance) {
      return b.importance - a.importance;
    }
    return new Date(b.created || 0) - new Date(a.created || 0);
  });
  
  // Handle memory form submission
  const handleMemorySubmit = async (e) => {
    e.preventDefault();
    
    if (activeView === 'add') {
      const success = await addMemory(newMemory);
      if (success) {
        setNewMemory({
          content: '',
          category: 'other',
          importance: 0.5
        });
        setActiveView('memories');
      }
    } else if (activeView === 'edit' && editingMemory) {
      // For editing, we'd need to implement an update API endpoint
      // For now, delete old and add new
      const deleteSuccess = await deleteMemory(editingMemory.content);
      if (deleteSuccess) {
        const addSuccess = await addMemory(newMemory);
        if (addSuccess) {
          setActiveView('memories');
          setEditingMemory(null);
        }
      }
    }
  };
  
  // Handle user profile form submission
  const handleProfileSubmit = (e) => {
    e.preventDefault();
    updateUserProfile(profileData);
    setActiveView('memories');
  };
  
  // Set up editing an existing memory
  const handleEditMemory = (memory) => {
    setEditingMemory(memory);
    setNewMemory({
      content: memory.content,
      category: memory.category,
      importance: memory.importance
    });
    setActiveView('edit');
  };
  
  // Format date string for display
  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch {
      return 'Unknown date';
    }
  };
  
  return (
    <div className="memory-editor">
      <div className="memory-editor-header">
        <h2>Memory Manager</h2>
        <div className="view-switcher">
          <button 
            className={activeView === 'memories' ? 'active' : ''} 
            onClick={() => setActiveView('memories')}
          >
            Memories
          </button>
          <button 
            className={activeView === 'profile' ? 'active' : ''} 
            onClick={() => setActiveView('profile')}
          >
            User Profile
          </button>
        </div>
        {onClose && (
          <button className="close-button" onClick={onClose}>✕</button>
        )}
      </div>
      
      {/* MEMORIES VIEW */}
      {activeView === 'memories' && (
        <div className="memories-view">
          {memoryError && (
            <div className="error-message" style={{color: 'red', margin: '1rem', padding: '0.5rem', background: '#ffe6e6', borderRadius: '4px'}}>
              Error loading memories: {memoryError}
              <button onClick={fetchMemories} style={{marginLeft: '1rem'}}>Retry</button>
            </div>
          )}
          
          <div className="filters-bar">
            <input 
              type="text" 
              placeholder="Search memories..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            
            <select 
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
            >
              <option value="">All Categories</option>
              {MEMORY_CATEGORIES.map(category => (
                <option key={category} value={category}>
                  {category.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </option>
              ))}
            </select>
            
            <button 
              className="add-button"
              onClick={() => setActiveView('add')}
            >
              Add Memory
            </button>
            
            <button onClick={fetchMemories} disabled={isLoadingMemories}>
              {isLoadingMemories ? 'Loading...' : 'Refresh'}
            </button>
          </div>
          
          {isLoadingMemories ? (
            <div style={{textAlign: 'center', padding: '2rem'}}>Loading memories...</div>
          ) : (
            <div className="memories-list">
              {sortedMemories.length > 0 ? (
                sortedMemories.map((memory, index) => (
                  <div key={`${memory.content}-${index}`} className="memory-card">
                    <div className="memory-content">
                      <p>{memory.content}</p>
                    </div>
                    
                    <div className="memory-meta">
                      <span className="memory-category">
                        {(memory.category || 'other').replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                      </span>
                      <span className="memory-importance" title={`Importance: ${memory.importance || 0.5}`}>
                        {Array.from({ length: Math.ceil((memory.importance || 0.5) * 5) }).map((_, i) => (
                          <span key={i}>★</span>
                        ))}
                      </span>
                      <span className="memory-type">
                        {memory.type === 'manual' ? 'Manual' : 'AI Generated'}
                      </span>
                    </div>
                    
                    <div className="memory-dates">
                      <div>Created: {formatDate(memory.created)}</div>
                      {memory.last_accessed && <div>Last accessed: {formatDate(memory.last_accessed)}</div>}
                      {memory.accessed && <div>Access count: {memory.accessed}</div>}
                    </div>
                    
                    <div className="memory-actions">
                      <button 
                        className="edit-button"
                        onClick={() => handleEditMemory(memory)}
                      >
                        Edit
                      </button>
                      <button 
                        className="delete-button"
                        onClick={() => deleteMemory(memory.content)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="no-memories">
                  <p>No memories found. Add your first memory to get started!</p>
                </div>
              )}
            </div>
          )}
          
          <div className="memories-actions">
            <button 
              className="reset-button"
              onClick={resetMemories}
              disabled={!activeProfileId}
            >
              Reset All Memories
            </button>
          </div>
        </div>
      )}
      
      {/* ADD MEMORY VIEW */}
      {activeView === 'add' && (
        <div className="add-memory-view">
          <h3>Add New Memory</h3>
          <form onSubmit={handleMemorySubmit}>
            <div className="form-group">
              <label htmlFor="content">Memory Content:</label>
              <textarea 
                id="content"
                value={newMemory.content}
                onChange={(e) => setNewMemory({...newMemory, content: e.target.value})}
                required
                rows={4}
                placeholder="What should the AI remember about you?"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="category">Category:</label>
              <select 
                id="category"
                value={newMemory.category}
                onChange={(e) => setNewMemory({...newMemory, category: e.target.value})}
              >
                {MEMORY_CATEGORIES.map(category => (
                  <option key={category} value={category}>
                    {category.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="importance">
                Importance: {newMemory.importance.toFixed(1)}
              </label>
              <input 
                type="range"
                id="importance"
                min="0"
                max="1"
                step="0.1"
                value={newMemory.importance}
                onChange={(e) => setNewMemory({...newMemory, importance: parseFloat(e.target.value)})}
              />
            </div>
            
            <div className="form-actions">
              <button type="submit">Save Memory</button>
              <button type="button" onClick={() => setActiveView('memories')}>
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}
      
      {/* EDIT MEMORY VIEW */}
      {activeView === 'edit' && editingMemory && (
        <div className="edit-memory-view">
          <h3>Edit Memory</h3>
          <form onSubmit={handleMemorySubmit}>
            <div className="form-group">
              <label htmlFor="content">Memory Content:</label>
              <textarea 
                id="content"
                value={newMemory.content}
                onChange={(e) => setNewMemory({...newMemory, content: e.target.value})}
                required
                rows={4}
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="category">Category:</label>
              <select 
                id="category"
                value={newMemory.category}
                onChange={(e) => setNewMemory({...newMemory, category: e.target.value})}
              >
                {MEMORY_CATEGORIES.map(category => (
                  <option key={category} value={category}>
                    {category.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="importance">
                Importance: {newMemory.importance.toFixed(1)}
              </label>
              <input 
                type="range"
                id="importance"
                min="0"
                max="1"
                step="0.1"
                value={newMemory.importance}
                onChange={(e) => setNewMemory({...newMemory, importance: parseFloat(e.target.value)})}
              />
            </div>
            
            <div className="form-actions">
              <button type="submit">Update Memory</button>
              <button type="button" onClick={() => setActiveView('memories')}>
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}
      
      {/* USER PROFILE VIEW */}
      {activeView === 'profile' && (
        <div className="profile-view">
          <ProfileSelector />
          
          <h3>Current Profile Settings</h3>
          <form onSubmit={handleProfileSubmit}>
            <div className="form-group">
              <label htmlFor="name">Name:</label>
              <input 
                type="text"
                id="name"
                value={profileData.name || ''}
                onChange={(e) => setProfileData({...profileData, name: e.target.value})}
                required
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="responseStyle">Preferred Response Style:</label>
              <select 
                id="responseStyle"
                value={profileData.preferences?.responseStyle || 'balanced'}
                onChange={(e) => setProfileData({
                  ...profileData, 
                  preferences: {
                    ...profileData.preferences,
                    responseStyle: e.target.value
                  }
                })}
              >
                <option value="concise">Concise</option>
                <option value="balanced">Balanced</option>
                <option value="detailed">Detailed</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="user-avatar">User Avatar:</label>
              <div className="flex items-center gap-4 mt-2">
                {profileData.avatar && (
                  <img
                    src={profileData.avatar.startsWith('/') ? `${MEMORY_API_URL}${profileData.avatar}` : profileData.avatar}
                    alt="User Avatar"
                    className="w-16 h-16 rounded-full object-cover border border-gray-300 dark:border-gray-600"
                    onError={(e) => { e.target.style.display = 'none'; console.warn('Failed to load avatar:', profileData.avatar); }}
                  />
               )}
              <Input
                id="user-avatar"
                type="file"
                accept="image/png, image/jpeg, image/gif, image/webp"
                onChange={handleUserAvatarUpload}
                disabled={isUploading}
                className="flex-grow"
              />
            </div>
            {isUploading && <p className="text-sm text-muted-foreground mt-1">Uploading...</p>}
          </div>
            
            <div className="form-group">
              <label>Interests/Topics:</label>
              <div className="topics-input">
                {profileData.preferences?.topics?.map((topic, index) => (
                  <div key={index} className="topic-tag">
                    {topic}
                    <button 
                      type="button"
                      onClick={() => {
                        const newTopics = [...(profileData.preferences?.topics || [])];
                        newTopics.splice(index, 1);
                        setProfileData({
                          ...profileData,
                          preferences: {
                            ...profileData.preferences,
                            topics: newTopics
                          }
                        });
                      }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
                <input 
                  type="text" 
                  placeholder="Add a topic and press Enter"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && e.target.value.trim()) {
                      e.preventDefault();
                      const newTopic = e.target.value.trim();
                      const currentTopics = profileData.preferences?.topics || [];
                      if (!currentTopics.includes(newTopic)) {
                        setProfileData({
                          ...profileData,
                          preferences: {
                            ...profileData.preferences,
                            topics: [...currentTopics, newTopic]
                          }
                        });
                      }
                      e.target.value = '';
                    }
                  }}
                />
              </div>
            </div>
            
            <div className="form-actions">
              <button type="submit">Save Profile</button>
              <button type="button" onClick={() => setActiveView('memories')}>
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};

export default MemoryEditor;